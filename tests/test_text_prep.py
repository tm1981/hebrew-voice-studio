"""Text preparation: how numbers, prices, dates, phone numbers and abbreviations become words.

Run from the project folder: python -m unittest discover tests
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_prep import Dictionary, normalize, prepare  # noqa: E402

# (text, what it should become)
CASES = [
    ('יש לי 5 ילדים.', 'יש לי חמישה ילדים.'),
    # Latin abbreviations inside Hebrew are written as the letters' Hebrew names
    ('ב-CNN דווח על כך.', 'בסי-אן-אן דווח על כך.'),
    ('סוכני ה-FBI הגיעו.', 'סוכני האף-בי-איי הגיעו.'),
    ('ערוצי BBC ו-CNN שידרו.', 'ערוצי בי-בי-סי וסי-אן-אן שידרו.'),
    ('לפי NASA, השיגור נדחה.', 'לפי נאסא, השיגור נדחה.'),
    ('The CNN report was long.', 'The CNN report was long.'),
    ('פרק II בספר', 'פרק II בספר'),
    ('אמר Claude Code', 'אמר Claude Code'),
    ('WOW איזה יופי', 'WOW איזה יופי'),
    # 13 to 24 after ב/מ/ל is the number itself, not an hour on a 12-hour clock ("22" was read "ten")
    ('תקלה נרחבת ב-22 מרכזי בקרת טיסה.', 'תקלה נרחבת בעשרים ושניים מרכזי בקרת טיסה.'),
    ('העלייה היא מ-22 עד 25.', 'העלייה היא מעשרים ושתיים עד עשרים וחמש.'),
    ('נפגשנו ב-14', 'נפגשנו בארבע עשרה'),
    ('ב־22 מרכזי בקרה', 'בעשרים ושניים מרכזי בקרה'),
    # with a word that says it is a time, the hour stays an hour, at both ends of a range
    ('הפגישה בשעה 22', 'הפגישה בשעה עשר'),
    ('בין השעות 14 ל-16', 'בין השעות שתיים לארבע'),
    ('נפגשנו ב-8 בערב', 'נפגשנו בשמונה בערב'),
    # a construct plural after a number is its noun
    ('ב-3 בתי ספר', 'בשלושה בתי ספר'),
    ('הצביעו 22 חברי כנסת.', 'הצביעו עשרים ושניים חברי כנסת.'),
    ('ב-8 אחרי הצהריים', 'בשמונה אחרי הצהריים'),
    ('יש לי 3 ילדים ו-2 בנות.', 'יש לי שלושה ילדים ושתי בנות.'),
    ('קניתי 12 ביצים ו-21 תפוחים.', 'קניתי שתים עשרה ביצים ועשרים ואחד תפוחים.'),
    ('בשנת 2024 עברנו דירה.', 'בשנת אלפיים עשרים וארבע עברנו דירה.'),
    ('ב-2000 גרנו בתל אביב.', 'באלפיים גרנו בתל אביב.'),
    ('נפגשנו ב-12/03/2024 בבוקר.', 'נפגשנו בשנים עשר במרץ אלפיים עשרים וארבע בבוקר.'),
    ('הפגישה ב-1.5.2023 בצהריים.', 'הפגישה באחד במאי אלפיים עשרים ושלוש בצהריים.'),
    ('ב-2024-03-12 נסענו לים.', 'בשנים עשר במרץ אלפיים עשרים וארבע נסענו לים.'),
    ('הרכבת יוצאת ב-15:30 מהתחנה.', 'הרכבת יוצאת בשלוש וחצי מהתחנה.'),
    ('המחיר ירד ב-15% השנה.', 'המחיר ירד בחמישה עשר אחוז השנה.'),
    ('בעיר גרים 2,500,000 אנשים.', 'בעיר גרים שני מיליון חמש מאות אלף אנשים.'),
    ('הגעתי למקום ה-3 בתחרות.', 'הגעתי למקום השלישי בתחרות.'),
    ('הוספתי 3/4 כוס חלב.', 'הוספתי שלושת רבעי כוס חלב.'),
    ('שנות ה-90 היו טובות.', 'שנות התשעים היו טובות.'),
    ('ד"ר כהן הגיע.', 'דוקטור כהן הגיע.'),
    ('קוד 250', 'קוד שתיים חמש אפס'),
    ('הכרטיס עולה 250 ₪.', 'הכרטיס עולה מאתיים וחמישים שקלים.'),
    ('קניתי כרטיס ב-100 ₪', 'קניתי כרטיס במאה שקלים'),
    ('הספר עולה 49.90 ₪.', 'הספר עולה ארבעים ותשעה שקלים ותשעים אגורות.'),
    ('עולה 0.50 ₪', 'עולה חמישים אגורות'),
    ('עולה 1 ₪', 'עולה שקל אחד'),
    ('עולה 2.01 ₪', 'עולה שני שקלים ואגורה אחת'),
    ('עולה $12.50', 'עולה שנים עשר דולר וחמישים סנט'),
    ('המחיר הוא 20$ בלבד.', 'המחיר הוא עשרים דולר בלבד.'),
    ('שילמתי 1,500 שקלים על המחשב.', 'שילמתי אלף וחמש מאות שקלים על המחשב.'),
    ('עולה 3 דולרים', 'עולה שלושה דולרים'),
    ('הדירה נמכרה ב-1.5 מיליון ₪.', 'הדירה נמכרה באחד וחצי מיליון שקלים.'),
    ('עולה 49.90 ש"ח', 'עולה ארבעים ותשעה שקלים ותשעים אגורות'),
    ('התקשרו 054-123-4567 היום.', 'התקשרו אפס חמש ארבע, אחת שתיים שלוש, ארבע חמש שש שבע היום.'),
    ('התקשרו ל-03-6123456.', 'התקשרו לאפס שלוש, שש אחת שתיים שלוש ארבע חמש שש.'),
    ('המספר +972-54-1234567', 'המספר פלוס תשע שבע שתיים, חמש ארבע, אחת שתיים שלוש ארבע חמש שש שבע'),
    ('Call 054-123-4567 now.', 'Call zero five four, one two three, four five six seven now.'),
    ('Call (555) 123-4567 today.', 'Call five five five, one two three, four five six seven today.'),
    ('My number is +972 54 123 4567.', 'My number is plus nine seven two, five four, one two three, four five six seven.'),
    ('Call +972541234567 now.', 'Call plus nine seven two five four, one two three, four five six seven now.'),
    ('اتصل على 054-123-4567 اليوم.', 'اتصل على صفر خمسة أربعة، واحد اثنان ثلاثة، أربعة خمسة ستة سبعة اليوم.'),
    ('Pages 10-20.', 'Pages ten-twenty.'),
    ('The score was 3-2-1.', 'The score was three-two-one.'),
    ('From 1990-2000 we lived there.', 'From nineteen ninety to two thousand we lived there.'),
    ('The war of 1939–1945.', 'The war of nineteen thirty-nine to nineteen forty-five.'),
    ('In 1990-95 we moved.', 'In nineteen ninety to nineteen ninety-five we moved.'),
    ('בשנים 1990-2000 גרנו שם.', 'בשנים אלף תשע מאות ותשעים עד אלפיים גרנו שם.'),
    ('ב-1990-2000 גרנו שם.', 'באלף תשע מאות ותשעים עד אלפיים גרנו שם.'),
    ('عاش هناك 1990-2000 سنوات.', 'عاش هناك ألف و تسعمائة و تسعون إلى ألفان سنوات.'),
    ('I have 5 kids.', 'I have five kids.'),
    ('It costs $20.', 'It costs twenty dollars.'),
    ('Born in 1990.', 'Born in nineteen ninety.'),
    ('The M4 chip has 16GB of memory.', 'The M four chip has sixteen GB of memory.'),
    ('It runs at 3.5GHz.', 'It runs at three point five GHz.'),
    ('We open at 10am and close at 5pm.', 'We open at ten A M and close at five P M.'),
    ('Open 24/7.', 'Open twenty-four seven.'),
    ('A 50/50 chance.', 'A fifty fifty chance.'),
    ('Add 3/4 of a cup.', 'Add three quarters of a cup.'),
    ('Bake for 1 1/2 hours.', 'Bake for one and a half hours.'),
    ('Use a 5/8 inch bolt.', 'Use a five eighths inch bolt.'),
    ('Rated 8/10.', 'Rated eight over ten.'),
    ('on 3/4', 'on the third of April'),
    ('In the 1990s we had 2 cars.', 'In the nineteen nineties we had two cars.'),
    ('The 80s and the 2000s.', 'The eighties and the two thousands.'),
    ('1000s of people', 'thousands of people'),
    ('He came 1st and she came 2nd.', 'He came first and she came second.'),
    ('On 12/03/2024 I paid $20.', 'On the twelfth of March twenty twenty-four I paid twenty dollars.'),
    ('On 2024-03-12 I paid.', 'On the twelfth of March twenty twenty-four I paid.'),
    ('No. 5', 'No. five'),
    ('יש לי 3 apples', 'יש לי three apples'),
    ('Chapter 3: The End', 'Chapter three: The End'),
    ('2. Open the app.', 'two: Open the app.'),
    ('1. First item\n2. Second item\n3. Third item', 'one: First item.\ntwo: Second item.\nthree: Third item.'),
    ('ولد في عام 1990 واشترى 5 كتب بسعر 20$ يوم 12/03/2024.', 'ولد في عام ألف و تسعمائة و تسعون واشترى خمسة كتب بسعر عشرون دولار يوم اثنا عشر مارس ألفان و أربعة و عشرون.'),
    ('ولد عام 2000.', 'ولد عام ألفان.'),
    ('اشترى 200 كتاب', 'اشترى مئتا كتاب'),
    ('קניתי מחשב ב-2024. The M4 chip has 16GB of memory. יש לו 3 שנות אחריות.', 'קניתי מחשב באלפיים עשרים וארבע. The M four chip has sixteen GB of memory. יש לו שלוש שנות אחריות.'),
]


class NormalizeTest(unittest.TestCase):
    def test_cases(self):
        for text, expected in CASES:
            with self.subTest(text=text):
                self.assertEqual(normalize(text), expected)

    def test_text_without_digits_is_unchanged(self):
        for text in ("שלום עולם.", "Hello there.", "مرحبا بك."):
            self.assertEqual(normalize(text), text)

    def test_long_runs_stay_fast(self):
        # Patterns that backtrack badly took seconds on inputs like these (the request limit is 5000).
        for text in ("1." + " " * 5000 + "x", "1" * 5000, "+1 " * 1700, "0-" * 2500, "1990-" * 1000, "₪ " * 2500):
            start = time.perf_counter()
            normalize(text)
            self.assertLess(time.perf_counter() - start, 1.0, text[:20])


class DictionaryTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.dictionary = Dictionary(Path(self.dir.name) / "pronunciations.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_entry_matches_with_hebrew_prefix(self):
        self.dictionary.set("תאמר", "תָּאִמְר")
        self.assertEqual(prepare("ותאמר הגיע", self.dictionary), "ותָּאִמְר הגיע")

    def test_entry_beats_built_in_abbreviation(self):
        self.dictionary.set('ד"ר', "דוקטורית")
        self.assertEqual(prepare('ד"ר כהן', self.dictionary), "דוקטורית כהן")

    def test_normalize_can_be_turned_off(self):
        self.assertEqual(prepare("יש 5 ילדים", self.dictionary, normalize_text=False), "יש 5 ילדים")


if __name__ == "__main__":
    unittest.main()
