"""Самопроверка логики нарушений: python test_siz.py"""
from siz_demo import Checklist, Hold, IouTracker, in_polygon, parse_points, parse_zone, person_problems

person = (100, 100, 200, 400)
helmet = ("helmet", (130, 100, 170, 130))
vest = ("safety-vest", (120, 180, 180, 280))

assert person_problems(person, [helmet, vest], None, True, True) == []
assert person_problems(person, [vest], None, True, True) == ["no helmet"]
assert person_problems(person, [helmet], None, True, True) == ["no vest"]
# каска в руке на уровне пояса не считается надетой
assert person_problems(person, [("helmet", (130, 250, 170, 280)), vest], None, True, True) == ["no helmet"]
# модель без классов СИЗ проверяет только зону
assert person_problems(person, [], None, False, False) == []

zone = parse_zone("0,350;300,350;300,500;0,500")
assert in_polygon((150, 400), zone) and not in_polygon((150, 300), zone)
assert person_problems(person, [helmet, vest], zone, True, True) == ["danger zone"]

h = Hold(2.0)
assert not h.update(1, ["no helmet"], 0.0)
assert not h.update(1, ["no helmet"], 1.9)
assert h.update(1, ["no helmet"], 2.0)       # сработало через 2 с
assert not h.update(1, ["no helmet"], 5.0)   # повторно в том же эпизоде не шлём
assert not h.update(1, [], 6.0)              # надел каску — эпизод закончен
assert not h.update(1, ["no helmet"], 7.0)
assert h.update(1, ["no helmet"], 9.0)       # новый эпизод — новое уведомление
h.forget(set())
assert h.since == {} and h.fired == set()

t = IouTracker()
a, b = (0, 0, 100, 200), (300, 0, 400, 200)
assert t.update([a, b]) == [1, 2]
assert t.update([(305, 5, 405, 205), (5, 0, 105, 200)]) == [2, 1]  # порядок поменялся — номера те же
assert t.update([(600, 0, 700, 200)]) == [3]                        # новый человек — новый номер

pts = parse_points(["Лебёдка=0,0;100,0;100,100;0,100", "Двигатель=200,0;300,0;300,100"])
assert list(pts) == ["Лебёдка", "Двигатель"] and pts["Двигатель"][2] == (300.0, 100.0)
assert parse_points(None) == {}
c = Checklist(pts, 2.0)
assert c.update([(50, 50)], 0.0, 1.0) == []                 # 1 с у лебёдки — мало
assert c.update([(500, 500)], 1.0, 1.0) == []               # отошёл — время не теряется, но и не копится
assert c.update([(50, 50), (60, 60)], 2.0, 1.0) == ["Лебёдка"]  # двое у узла считаются за одного
assert c.update([(50, 50)], 3.0, 1.0) == []                 # выполненный пункт повторно не засчитывается
assert c.done == {"Лебёдка": 2.0} and c.missed() == ["Двигатель"]

print("OK: все проверки логики пройдены")
