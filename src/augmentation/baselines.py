"""
Comparative Baseline Augmenters: EDA (Easy Data Augmentation) and Back-Translation for Bahasa Indonesia.
Provides standardized baseline implementations to benchmark against causal transformer data augmentation.
"""

import random
import re
from typing import List, Dict, Optional, Tuple

# Extensive Indonesian Synonym Dictionary for EDA
INDONESIAN_SYNONYMS: Dict[str, List[str]] = {
    # Sentiment Positive
    "bagus": ["baik", "elok", "mantap", "hebat", "apik", "prima", "keren"],
    "baik": ["bagus", "elok", "ramah", "terpuji", "patut", "apik"],
    "keren": ["hebat", "mantap", "bagus", "menakjubkan", "luar biasa", "kece"],
    "senang": ["gembira", "bahagia", "suka", "lega", "girang", "puas"],
    "suka": ["gemar", "senang", "cinta", "tertarik", "menikmati"],
    "nyaman": ["enak", "tentram", "sedap", "adem", "tenang", "segar"],
    "enak": ["nyaman", "lezat", "nikmat", "sedap", "gurih"],
    "bersih": ["higienis", "rapi", "resik", "bening", "kinclong"],
    "ramah": ["hangat", "sopan", "santun", "baik", "bersahabat"],
    "puas": ["lega", "senang", "terpenuhi", "bangga"],
    "cepat": ["lekas", "tangkas", "gesit", "kilat", "segera"],
    "luas": ["lapang", "lebar", "lega", "besar"],
    "lengkap": ["komplit", "sempurna", "penuh", "utuh"],
    "murah": ["terjangkau", "miring", "ekonomis"],

    # Sentiment Negative
    "buruk": ["jelek", "rusak", "parah", "bobrok", "mengecewakan"],
    "jelek": ["buruk", "cacat", "rusak", "suram"],
    "kecewa": ["sesal", "masygul", "putus asa", "patah hati"],
    "sedih": ["duka", "pilu", "susah", "pedih", "murung"],
    "marah": ["kesal", "murka", "geram", "emosi", "dongkol"],
    "kesal": ["jengkel", "marah", "sebal", "mangkel"],
    "kotor": ["jorok", "kumuh", "dekil", "cemar", "bernoda"],
    "bau": ["amis", "busuk", "pengap", "tengik", "anyir"],
    "sempit": ["sesak", "terbatas", "ketat"],
    "bising": ["ribut", "gaduh", "ramai", "berisik"],
    "mahal": ["tinggi", "selangit", "kurang terjangkau"],
    "lambat": ["lelet", "pelan", "lamban"],
    "rusak": ["hancur", "bobrok", "cacat", "error"],
    "takut": ["khawatir", "gentar", "cemas", "ngeri"],
    "cemas": ["khawatir", "gelisah", "was-was", "takut"],

    # Modifiers & Neutral
    "sangat": ["amat", "sekali", "banget", "sungguh", "teramat"],
    "banget": ["sangat", "amat", "sekali", "sungguh"],
    "agak": ["sedikit", "kurang", "kira-kira"],
    "cukup": ["lumayan", "sedang", "memadai"],
    "hanya": ["cuma", "semata", "doang"],
    "bisa": ["dapat", "mampu", "sanggup"],
    "tidak": ["tak", "tiada", "bukan"],
    "selalu": ["senantiasa", "kerap", "sering"],
    "sering": ["kerap", "seringkali", "banyak"],
    "kamar": ["ruangan", "bilik", "room"],
    "hotel": ["penginapan", "akomodasi", "losmen"],
    "pelayanan": ["layanan", "service", "penanganan"],
    "makanan": ["santapan", "kuliner", "hidangan", "sajian"],
    "film": ["sinema", "layar lebar", "tontonan", "movie"],
    "tempat": ["lokasi", "area", "spot"],
    "staf": ["karyawan", "pegawai", "pelayan", "petugas"],
}

INDONESIAN_STOPWORDS = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua", "ia",
    "seperti", "jika", "sehingga", "kembali", "dan", "ini", "karena", "oleh", "saat", "oleh",
    "setelah", "kurang", "hanya", "dari", "dalam", "bisa", "adalah", "di", "itu", "dengan"
}


class IndonesianEDA:
    """Easy Data Augmentation (EDA) implementation tailored for Bahasa Indonesia."""

    def __init__(
        self,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
        num_aug: int = 1,
        seed: Optional[int] = 42,
    ):
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug
        if seed is not None:
            random.seed(seed)

    def get_synonyms(self, word: str) -> List[str]:
        word_lower = word.lower()
        return INDONESIAN_SYNONYMS.get(word_lower, [])

    def synonym_replacement(self, words: List[str], n: int) -> List[str]:
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.lower() not in INDONESIAN_STOPWORDS]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return new_words

    def random_insertion(self, words: List[str], n: int) -> List[str]:
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words: List[str]) -> None:
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            if not new_words or counter >= 10:
                return
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1

        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words))
        new_words.insert(random_idx, random_synonym)

    def random_swap(self, words: List[str], n: int) -> List[str]:
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words: List[str]) -> List[str]:
        if len(new_words) < 2:
            return new_words
        idx1 = random.randint(0, len(new_words) - 1)
        idx2 = idx1
        counter = 0
        while idx2 == idx1:
            idx2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def random_deletion(self, words: List[str], p: float) -> List[str]:
        if len(words) <= 2:
            return words
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]
        return new_words

    def augment_sentence(self, sentence: str) -> str:
        words = sentence.split()
        num_words = len(words)
        if num_words <= 2:
            return sentence

        n_sr = max(1, int(self.alpha_sr * num_words))
        n_ri = max(1, int(self.alpha_ri * num_words))
        n_rs = max(1, int(self.alpha_rs * num_words))

        op = random.choice(["sr", "ri", "rs", "rd"])
        if op == "sr":
            a_words = self.synonym_replacement(words, n_sr)
        elif op == "ri":
            a_words = self.random_insertion(words, n_ri)
        elif op == "rs":
            a_words = self.random_swap(words, n_rs)
        else:
            a_words = self.random_deletion(words, self.p_rd)

        return " ".join(a_words)

    def augment(self, sentence: str, n: int = 1) -> List[str]:
        """Generates n EDA augmented variants of the given sentence."""
        return [self.augment_sentence(sentence) for _ in range(n)]


class IndonesianBackTranslator:
    """Back-translation paraphrasing pipeline for Bahasa Indonesia."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._id2en_model = None
        self._id2en_tok = None
        self._en2id_model = None
        self._en2id_tok = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            m_id2en = "Helsinki-NLP/opus-mt-id-en"
            m_en2id = "Helsinki-NLP/opus-mt-en-id"
            self._id2en_tok = MarianTokenizer.from_pretrained(m_id2en)
            self._id2en_model = MarianMTModel.from_pretrained(m_id2en).to(self.device)
            self._en2id_tok = MarianTokenizer.from_pretrained(m_en2id)
            self._en2id_model = MarianMTModel.from_pretrained(m_en2id).to(self.device)
            self._initialized = True
        except Exception:
            self._initialized = False

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        self._lazy_init()
        if not self._initialized:
            results = []
            for text in texts:
                words = text.split()
                out = []
                for w in words:
                    syns = INDONESIAN_SYNONYMS.get(w.lower())
                    if syns and random.random() < 0.35:
                        out.append(random.choice(syns))
                    else:
                        out.append(w)
                results.append(" ".join(out))
            return results

        import torch
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._id2en_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                translated_en = self._id2en_model.generate(**inputs, max_length=128)
                en_texts = self._id2en_tok.batch_decode(translated_en, skip_special_tokens=True)

                inputs_en = self._en2id_tok(en_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                translated_id = self._en2id_model.generate(**inputs_en, max_length=128)
                id_texts = self._en2id_tok.batch_decode(translated_id, skip_special_tokens=True)
                results.extend(id_texts)

        return results
