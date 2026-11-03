"""Прототип: контроль СИЗ, опасных зон и ежедневного ТО оборудования по видео.

Запуск: python siz_demo.py --source video.mp4 --model ppe.pt --zone "100,400;600,400;600,700;100,700" --save out.mp4
ТО: добавь --point "Лебёдка=x,y;x,y;x,y" --point "Двигатель=..." — узлы, у которых бригада должна побывать.
"""
import argparse
import os
import time
from pathlib import Path

HELMET = {"helmet", "hardhat", "hard-hat"}
VEST = {"vest", "safety-vest", "safety vest"}
RU = {"no helmet": "без каски", "no vest": "без жилета", "danger zone": "в опасной зоне"}


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def in_polygon(pt, poly):
    x, y = pt
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def feet(b):
    return (b[0] + b[2]) / 2, b[3]


def person_problems(person, items, zone, check_helmet, check_vest):
    """person — рамка (x1,y1,x2,y2), items — [(имя класса, рамка)] всех детекций кадра."""
    x1, y1, x2, y2 = person
    head = (x1, y1, x2, y1 + 0.35 * (y2 - y1))  # каска ищется в верхней трети рамки человека
    problems = []
    if check_helmet and not any(n in HELMET and inside(center(b), head) for n, b in items):
        problems.append("no helmet")
    if check_vest and not any(n in VEST and inside(center(b), person) for n, b in items):
        problems.append("no vest")
    if zone and in_polygon(feet(person), zone):
        problems.append("danger zone")
    return problems


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter or 1)


class IouTracker:
    """Номера людей между кадрами по перекрытию рамок. СИЗ не трекаем: трекер YOLO теряет их при смене сцены.
    shortcut: человек, пропавший на 1 кадр, получает новый номер; для продукта — трекер с памятью (ByteTrack по людям)."""

    def __init__(self, thr=0.3):
        self.thr, self.prev, self.next_id = thr, {}, 1

    def update(self, boxes):
        cur, ids = {}, []
        for b in boxes:
            score, tid = max(((iou(b, pb), t) for t, pb in self.prev.items() if t not in cur), default=(0, None))
            if score < self.thr:
                tid, self.next_id = self.next_id, self.next_id + 1
            cur[tid] = b
            ids.append(tid)
        self.prev = cur
        return ids


class Hold:
    """Нарушение засчитывается, если держится не меньше seconds; одно уведомление на эпизод."""

    def __init__(self, seconds):
        self.seconds, self.since, self.fired = seconds, {}, set()

    def update(self, track_id, problems, now):
        if not problems:
            self.since.pop(track_id, None)
            self.fired.discard(track_id)
            return False
        start = self.since.setdefault(track_id, now)
        if now - start >= self.seconds and track_id not in self.fired:
            self.fired.add(track_id)
            return True
        return False

    def forget(self, active_ids):
        for tid in set(self.since) - active_ids:
            self.since.pop(tid)
            self.fired.discard(tid)


class Checklist:
    """Ежедневное ТО: пункт выполнен, если у узла кто-то простоял суммарно не меньше seconds.
    shortcut: засчитываем присутствие у узла, а не само действие (долил масло, осмотрел трос);
    для продукта — распознавание действий по позе рук или фото-подтверждение пункта с телефона."""

    def __init__(self, points, seconds):
        self.points, self.seconds = points, seconds
        self.spent = {n: 0.0 for n in points}
        self.done = {}

    def update(self, feet_pts, now, dt):
        """feet_pts — точки ног людей в кадре; возвращает пункты, выполненные на этом кадре."""
        new = []
        for name, poly in self.points.items():
            if name not in self.done and any(in_polygon(f, poly) for f in feet_pts):
                self.spent[name] += dt
                if self.spent[name] >= self.seconds:
                    self.done[name] = now
                    new.append(name)
        return new

    def missed(self):
        return [n for n in self.points if n not in self.done]


def parse_zone(s):
    return [tuple(map(float, p.split(","))) for p in s.split(";")] if s else None


def parse_points(items):
    return dict((name, parse_zone(poly)) for name, poly in (s.split("=", 1) for s in items or []))


def blur_faces(img, cv2, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # shortcut: Haar пропускает лица в профиль и в масках; для продукта — нейросетевой детектор лиц (YuNet)
    for x, y, w, h in cascade.detectMultiScale(gray, 1.1, 3):  # 3, а не 5: лишнее размытие лучше пропущенного лица
        img[y:y + h, x:x + w] = cv2.GaussianBlur(img[y:y + h, x:x + w], (51, 51), 0)
    return img


def send_telegram(jpg, caption):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data={"chat_id": chat, "caption": caption},
                      files={"photo": ("v.jpg", jpg)}, timeout=10).raise_for_status()
    except requests.RequestException as e:
        print("Telegram: не отправлено:", e)


def main():
    ap = argparse.ArgumentParser(description="Контроль СИЗ и опасных зон по видео")
    ap.add_argument("--source", default="0", help="файл, номер веб-камеры или rtsp://...")
    ap.add_argument("--model", default="yolov8n.pt", help="веса YOLO; для касок/жилетов нужна модель СИЗ")
    ap.add_argument("--zone", help='опасная зона в пикселях: "x,y;x,y;x,y"')
    ap.add_argument("--hold", type=float, default=2.0, help="сколько секунд должно длиться нарушение")
    ap.add_argument("--point", action="append", help='узел для ТО: "Название=x,y;x,y;x,y", можно несколько раз')
    ap.add_argument("--dwell", type=float, default=5.0, help="сколько секунд нужно пробыть у узла, чтобы пункт ТО засчитался")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default="violations", help="папка для кадров нарушений")
    ap.add_argument("--save", help="записать размеченное видео, напр. out.mp4")
    ap.add_argument("--show", action="store_true", help="показывать окно (q — выход)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    names = {i: n.lower() for i, n in model.names.items()}
    check_helmet = bool(HELMET & set(names.values()))
    check_vest = bool(VEST & set(names.values()))
    print(f"Классы модели: каска={'да' if check_helmet else 'нет'}, жилет={'да' if check_vest else 'нет'}")
    zone = parse_zone(args.zone)
    points = parse_points(args.point)
    checklist = Checklist(points, args.dwell) if points else None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    Path(args.out).mkdir(exist_ok=True)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть источник: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer, hold, tracker, i, events, t0 = None, Hold(args.hold), IouTracker(), 0, 0, time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = model.predict(frame, conf=args.conf, verbose=False)[0].boxes
        dets = [(names[int(c)], b) for c, b in zip(boxes.cls.tolist(), boxes.xyxy.tolist())]
        people = [b for n, b in dets if n == "person"]
        clean = frame.copy()
        active, fired = set(), []

        for b, tid in zip(people, tracker.update(people)):
            problems = person_problems(b, dets, zone, check_helmet, check_vest)
            color = (0, 0, 255) if problems else (0, 200, 0)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid} " + ", ".join(problems), (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            active.add(tid)
            if hold.update(tid, problems, i / fps):
                fired.append((tid, problems, (x1, y1, x2, y2)))
        hold.forget(active)
        import numpy as np
        if zone:
            cv2.polylines(frame, [np.array(zone, np.int32)], True, (0, 165, 255), 2)
        if checklist:
            for name in checklist.update([feet(b) for b in people], i / fps, 1 / fps):
                print(f"[{i / fps:7.1f} с] ТО: пункт «{name}» выполнен")
            # Hershey-шрифт OpenCV не рисует кириллицу, поэтому на кадре номера узлов, а названия — в отчёте
            for k, (name, poly) in enumerate(points.items(), 1):
                color = (0, 200, 0) if name in checklist.done else (255, 128, 0)
                cv2.polylines(frame, [np.array(poly, np.int32)], True, color, 2)
                cv2.putText(frame, f"T{k}" + (" OK" if name in checklist.done else ""), tuple(map(int, poly[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for tid, problems, (x1, y1, x2, y2) in fired:
            events += 1
            snap = blur_faces(clean.copy(), cv2, cascade)
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            path = Path(args.out) / f"{time.strftime('%Y%m%d_%H%M%S')}_id{tid}.jpg"
            cv2.imwrite(str(path), snap)
            caption = f"Нарушение (человек #{tid}): {', '.join(RU[p] for p in problems)}"
            print(f"[{i / fps:7.1f} с] {caption} -> {path}")
            send_telegram(cv2.imencode(".jpg", snap)[1].tobytes(), caption)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if args.show:
            cv2.imshow("SIZ", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    dt = time.monotonic() - t0
    print(f"Кадров: {i}, нарушений: {events}, скорость: {i / dt:.1f} кадр/с")

    if checklist:
        # shortcut: отчёт пишется в конце видео; для продукта — по расписанию смены (например, к 9:00)
        missed = checklist.missed()
        lines = [f"ТО: выполнено {len(checklist.done)} из {len(points)}"]
        lines += [f"  + {n} — на {t:.0f} с" for n, t in checklist.done.items()]
        lines += [f"  - {n} — НЕ выполнено" for n in missed]
        report = "\n".join(lines)
        print(report)
        path = Path(args.out) / f"{time.strftime('%Y%m%d_%H%M%S')}_to.txt"
        path.write_text(report, encoding="utf-8")
        if missed and i:
            send_telegram(cv2.imencode(".jpg", blur_faces(clean, cv2, cascade))[1].tobytes(), report)


if __name__ == "__main__":
    main()
