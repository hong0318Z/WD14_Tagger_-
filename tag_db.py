import csv
import io
import os


class TagDB:
    """Loads the user-provided danbooru tag CSV.

    Expected columns (no header): name, category, count, description
    where description is a Korean string that may end with " / 키워드: a, b, c"
    """

    def __init__(self):
        self.by_name = {}
        self.sources = []  # absolute paths of every CSV file currently merged in

    def load(self, file_obj):
        """Load a single CSV, replacing any previously loaded data. For loading
        several files at once (kept resident together), use load_many()."""
        self.by_name = {}
        self.sources = []
        if file_obj is None:
            return 0
        return self._merge_one(file_obj)

    def load_many(self, file_objs):
        """Load and merge multiple CSV files into one resident DB. Later files'
        entries win on name conflicts. Replaces any previously loaded data."""
        self.by_name = {}
        self.sources = []
        total = 0
        for file_obj in file_objs or []:
            total = self._merge_one(file_obj)
        return total

    def _merge_one(self, file_obj):
        path = None
        if isinstance(file_obj, str) and os.path.exists(file_obj):
            path = file_obj
            f = open(file_obj, "r", encoding="utf-8-sig", newline="")
        elif hasattr(file_obj, "name"):
            path = file_obj.name
            f = open(file_obj.name, "r", encoding="utf-8-sig", newline="")
        else:
            f = io.StringIO(file_obj)

        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            name, category, count, description = row[0], row[1], row[2], row[3]
            name = name.strip()
            if not name:
                continue
            keywords = []
            if " / 키워드: " in description:
                _, kw = description.split(" / 키워드: ", 1)
                keywords = [k.strip() for k in kw.split(",") if k.strip()]
            self.by_name[name] = {
                "name": name,
                "category": category,
                "count": count,
                "description": description,
                "keywords": keywords,
            }

        f.close()
        if path and path not in self.sources:
            self.sources.append(os.path.abspath(path))
        return len(self.by_name)

    def __len__(self):
        return len(self.by_name)

    def exists(self, tag_name: str) -> bool:
        return tag_name in self.by_name

    def filter_existing(self, tags):
        """tags: list of (name, prob) -> keeps only tags present in the DB."""
        return [(n, p) for n, p in tags if n in self.by_name]

    def search(self, query: str, limit: int = 15):
        """Search tag names / korean keywords / description for a query string.
        Returns a list of dicts with name, korean_name(=keywords), description.
        """
        query = query.strip().lower()
        if not query:
            return []
        words = [w for w in query.split() if len(w) >= 3]
        results = []
        for entry in self.by_name.values():
            haystack = entry["name"].lower().replace("_", " ")
            in_keywords = any(query in k.lower() for k in entry["keywords"])
            in_desc = query in entry["description"].lower()
            match = query in haystack or in_keywords or in_desc
            if not match and words:
                match = any(
                    w in haystack
                    or any(w in k.lower() for k in entry["keywords"])
                    for w in words
                )
            if match:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def candidates_for_terms(self, terms, per_term_limit: int = 8):
        """Given a list of English/Korean concept terms, return a deduped list
        of candidate tag dicts found in the DB."""
        seen = set()
        candidates = []
        for term in terms:
            for entry in self.search(term, limit=per_term_limit):
                if entry["name"] not in seen:
                    seen.add(entry["name"])
                    candidates.append(entry)
        return candidates
