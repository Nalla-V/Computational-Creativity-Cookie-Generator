import json
import random
import copy
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from itertools import combinations


# Prepare NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


STOPWORDS = set(stopwords.words('english')) | {
    'g', 'ml', 'half', 'an', 'the', 'and', 'all-purpose',
    'oil', 'powdered', 'flour', 'extract', 'grated', 'diced', 'rolled',
    'baking', 'yolk', 'egg', 'oat', 'mashed', 'powder', 'flakes', 'sugar',
    'salt', 'cake', 'puree', 'ghee', 'seeds', 'mix', 'soda', 'of', 'to', 'for',
    'cup', 'tsp', 'tbsp', 'unit', 'piece', 'chopped', 'toasted', 'bitter',
    'instant', 'mild', 'white', 'sea', 'dried', 'paste', 'juice', 'water'
}


model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_token(text):
    # Remove text in parentheses, trim and lowercase
    text = re.sub(r'\(.*?\)', '', text).strip().lower()
    return text


def filter_stopwords(words):
    # Remove stopwords and very short words, capitalize remaining
    return [w.capitalize() for w in words if w not in STOPWORDS and len(w) > 1]


def recipe_to_text(recipe):
    # Concatenate ingredient description strings from all sections
    parts = []
    for section in ['base_ingredients', 'flavour_addins', 'external_toppings']:
        for ing in recipe.get(section, []):
            parts.append(f"{ing['amount']} {ing['unit']} {ing['ingredient']}")
    return ", ".join(parts)


class CookieRecipe:
    def __init__(self, data):
        self.name = data.get("name", "Unnamed Cookie")
        self.base_ingredients = copy.deepcopy(data.get("base_ingredients", []))
        self.flavour_addins = copy.deepcopy(data.get("flavour_addins", []))
        self.external_toppings = copy.deepcopy(data.get("external_toppings", []))
        self.merge_ingredients()

    def all_ingredients(self):
        return self.base_ingredients + self.flavour_addins + self.external_toppings

    def merge_ingredients(self):
        # Merge duplicate ingredients in each section and limit toppings to three
        for section_name in ["base_ingredients", "flavour_addins", "external_toppings"]:
            section = getattr(self, section_name)
            merged = {}
            for ing in section:
                key = (ing['ingredient'].lower(), ing['unit'], ing.get('role'), ing.get('type'))
                if key in merged:
                    merged[key]['amount'] += ing['amount']
                else:
                    merged[key] = copy.deepcopy(ing)
            merged_list = list(merged.values())
            if section_name == "external_toppings":
                merged_list.sort(key=lambda x: x['amount'], reverse=True)
                merged_list = merged_list[:3]
            setattr(self, section_name, merged_list)

    def mutate(self, ingredient_pool, population=None, max_tries=5):
        tries = 0
        while tries < max_tries:
            tries += 1
            mutation_type = random.choice(["add", "remove", "replace", "adjust_amount"])
            section_choice = random.choice(["base_ingredients", "flavour_addins", "external_toppings"])
            section = getattr(self, section_choice)
            pool = ingredient_pool[section_choice]
            if not pool:
                return

            if mutation_type == "add" and section_choice == "flavour_addins" and population:
                all_flavour_names = [clean_token(i["ingredient"]) for r in population for i in r.flavour_addins]
                if all_flavour_names:
                    freq = {name: all_flavour_names.count(name) for name in all_flavour_names}
                    rare_pool = [i for i in pool if freq.get(clean_token(i["ingredient"]), 0) < 1]
                    if rare_pool:
                        section.append(copy.deepcopy(random.choice(rare_pool)))
                        self.merge_ingredients()
                        return

            if mutation_type == "add":
                section.append(copy.deepcopy(random.choice(pool)))
            elif mutation_type == "remove" and len(section) > 1:
                section.remove(random.choice(section))
            elif mutation_type == "replace" and section:
                ing = random.choice(section)
                ing.update(copy.deepcopy(random.choice(pool)))
            elif mutation_type == "adjust_amount" and section:
                ing = random.choice(section)
                ing["amount"] = round(ing["amount"] * random.uniform(0.8, 1.3), 2)
            self.merge_ingredients()
            return

    def crossover(self, other):
        child = copy.deepcopy(self)
        for section_name in ["base_ingredients", "flavour_addins", "external_toppings"]:
            a, b = getattr(self, section_name), getattr(other, section_name)
            if len(a) > 1 and len(b) > 1:
                if random.random() < 0.3:
                    child_section = copy.deepcopy(b)
                else:
                    cut = random.randint(1, min(len(a), len(b)) - 1)
                    child_section = a[:cut] + b[cut:]
            else:
                child_section = a + b
            setattr(child, section_name, child_section)
        child.merge_ingredients()
        child.name = child.generate_name()
        return child

    def generate_name(self):
        def clean_word(w):
            w = re.sub(r'[^a-zA-Z ]', '', w).strip().lower()
            w = re.sub(r'\b(halves?|flakes?|crushed|toasted|ground|grated|powder|paste|fresh|dried|sweetened|unsweetened|roasted|chopped|minced|sliced|pieces?|bits?)\b', '', w)
            w = w.strip()
            if not w or w in STOPWORDS or w in {'cookie', 'dough', 'batter', 'base'}:
                return None
            return w.capitalize()

        all_candidates = []
        for section in [self.flavour_addins, self.external_toppings]:
            for ing in section:
                words = [clean_word(w) for w in ing['ingredient'].split()]
                words = [w for w in words if w]  # remove None
                if not words:
                    continue
                main_word = words[-1] if len(words) > 1 else words[0]
                if main_word and main_word not in STOPWORDS:
                    all_candidates.append((main_word, ing['amount']))

        # Sort by descending ingredient amount
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        # Choose top 3 unique, non-stopword flavor words
        used = []
        for word, _ in all_candidates:
            if word not in used and len(used) < 3:
                used.append(word)

        # Check if any adjective-like descriptor (optional aesthetic)
        adjectives = {"Sweet", "Spicy", "Dark", "Toffee", "Golden", "Caramel", "Zesty"}
        descriptor = None
        for w in used:
            if w in adjectives:
                descriptor = w
                used.remove(w)
                break

        if descriptor:
            name_parts = [descriptor] + used
        else:
            name_parts = used

        name = " ".join(name_parts[:3]) + " Cookie"
        return name.strip()

    def to_text(self):
        return recipe_to_text(self.__dict__)

    def get_major_flavours(self):
        return set(clean_token(ing['ingredient']) for ing in self.flavour_addins if clean_token(ing['ingredient']))

    def __repr__(self):
        return f"{self.name}"


class CookieGeneticAlgorithm:
    def __init__(self, recipes, mutation_rate=0.6, population_size=20):
        self.recipes = [CookieRecipe(r) for r in recipes]
        self.mutation_rate = mutation_rate
        self.population_size = population_size

        self.ingredient_pool = {
            "base_ingredients": [i for r in recipes for i in r["base_ingredients"]],
            "flavour_addins": [i for r in recipes for i in r["flavour_addins"]],
            "external_toppings": [i for r in recipes for i in r["external_toppings"]],
        }

        self.original_embeddings = model.encode([recipe_to_text(r) for r in recipes])

    def evaluate_fitness(self, recipe):
        score = 0.0
        all_ing = recipe.all_ingredients()
        has_structure = any(i.get("role") == "structure" for i in all_ing)
        has_fat = any(i.get("role") == "fat" for i in all_ing)
        has_leavening = any(i.get("role") == "leavening" for i in all_ing)
        has_binder = any(i.get("role") == "binder" for i in all_ing)
        if not (has_structure and has_fat and has_leavening and has_binder):
            return -10   # Heavy penalty to miss the base ingredients

        uniq_count = len(set(clean_token(i["ingredient"]) for i in all_ing if clean_token(i["ingredient"])))
        score += uniq_count * 0.6

        base, addin, topping = len(recipe.base_ingredients), len(recipe.flavour_addins), len(recipe.external_toppings)
        score -= abs(base - 3) * 0.3
        if 2 <= addin <= 6:
            score += 1
        if topping > 0:
            score += 0.5

        text = recipe_to_text(recipe.__dict__)
        emb = model.encode([text])
        sim = cosine_similarity(emb, self.original_embeddings).max()
        novelty = 1 - sim
        score += novelty * 4  # Rewarding novelty to create new receipes

        return round(score, 2)

    def evolve_population(self, generations=8, max_attempts=200):
        num_originals = self.population_size // 3
        pool_originals = random.sample(self.recipes, num_originals)

        pool_mutated = []
        max_mutate_tries = 5
        while len(pool_mutated) < self.population_size - num_originals:
            r = copy.deepcopy(random.choice(self.recipes))
            mutate_tries = 0
            while mutate_tries < max_mutate_tries:
                r.mutate(self.ingredient_pool)
                mutate_tries += 1
            pool_mutated.append(r)

        population = pool_originals + pool_mutated

        total_generated = 0
        for _ in range(generations):
            scores = [self.evaluate_fitness(r) for r in population]
            new_pop = []
            attempts = 0
            while len(new_pop) < self.population_size and attempts < max_attempts * self.population_size:
                attempts += 1
                total_generated += 1
                parents = random.choices(population, weights=[max(s, 0.1) for s in scores], k=2)
                child = parents[0].crossover(parents[1])
                if random.random() < self.mutation_rate:
                    for _ in range(random.randint(1, 2)):
                        child.mutate(self.ingredient_pool, population)
                child.merge_ingredients()
                new_pop.append(child)
            population = new_pop

        fitnesses = [self.evaluate_fitness(r) for r in population]
        count_high = sum(1 for f in fitnesses if f >= 7)
        count_medium = sum(1 for f in fitnesses if 3 <= f < 7)
        count_low = sum(1 for f in fitnesses if f < 3)

        print(f"Total generated recipes this evolution: {total_generated}")
        print(f"Recipes with fitness >= 7: {count_high}")
        print(f"Recipes with fitness 3-7: {count_medium}")
        print(f"Recipes with fitness < 3: {count_low}")

        return population

    def select_three_distinct(self, population, fitness_threshold=3):
        texts = [r.to_text() for r in population]
        embeddings = model.encode(texts)
        fitnesses = [self.evaluate_fitness(r) for r in population]

        filtered = [(i, r, f) for i, (r, f) in enumerate(zip(population, fitnesses)) if f >= fitness_threshold]
        if not filtered:
            print("No recipes meet fitness threshold, lowering threshold.")
            filtered = list(enumerate(population))

        indices, filtered_recipes, filtered_fitnesses = zip(*filtered)
        idx_sorted = sorted(range(len(filtered_recipes)), key=lambda i: filtered_fitnesses[i], reverse=True)
        selected_indices = [idx_sorted[0]]

        for _ in range(2):
            remaining = [i for i in range(len(filtered_recipes)) if i not in selected_indices]
            max_dist = -1
            best_idx = None
            for idx in remaining:
                dists = [1 - cosine_similarity(embeddings[indices[idx]].reshape(1, -1), embeddings[indices[s]].reshape(1, -1))[0][0] for s in selected_indices]
                min_dist = min(dists)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_idx = idx
            if best_idx is not None:
                selected_indices.append(best_idx)

        return [filtered_recipes[i] for i in selected_indices]


def visualize_recipe_space(original_recipes, generated_recipes):
    orig_texts = [recipe_to_text(r) for r in original_recipes]
    gen_texts = [r.to_text() for r in generated_recipes]
    orig_vecs = model.encode(orig_texts)
    gen_vecs = model.encode(gen_texts)
    all_vecs = np.vstack([orig_vecs, gen_vecs])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_vecs)

    n_orig = len(orig_texts)
    plt.figure(figsize=(9, 7))
    plt.scatter(reduced[:n_orig, 0], reduced[:n_orig, 1], c='blue', label='Original Recipes', s=70)
    plt.scatter(reduced[n_orig:, 0], reduced[n_orig:, 1], c='red', label='Generated Cookies', s=90, marker='x')
    plt.legend()
    plt.title("Recipe Embedding Space")
    plt.xlabel("Principal Component 1")  
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.savefig('embedding_space.png', dpi=300)
    plt.show()

def print_report_metrics(original_population, selected_recipes, ga_instance):
    print("\n--- Report Metrics ---\n")

    # Population fitness distribution
    fitnesses = [ga_instance.evaluate_fitness(r) for r in original_population]
    total = len(fitnesses)
    count_high = sum(1 for f in fitnesses if f >= 7)
    count_medium = sum(1 for f in fitnesses if 3 <= f < 7)
    count_low = sum(1 for f in fitnesses if f < 3)

    print("Population Fitness Distribution:")
    print(f" Fitness >= 7: {count_high} ({count_high / total:.1%})")
    print(f" Fitness 3 - 7: {count_medium} ({count_medium / total:.1%})")
    print(f" Fitness < 3: {count_low} ({count_low / total:.1%})\n")

    # Summary of top recipes
    print(f"Top {len(selected_recipes)} Recipes Summary:")
    print("| Rank | Name                             | Fitness | Base Ing. | Flavour Addins | Toppings | Major Flavours               |")
    print("|------|----------------------------------|---------|-----------|----------------|----------|-----------------------------|")
    for i, r in enumerate(selected_recipes, 1):
        fitness = ga_instance.evaluate_fitness(r)
        base_count = len(r.base_ingredients)
        addin_count = len(r.flavour_addins)
        topping_count = len(r.external_toppings)
        major_flavours = ", ".join(sorted(r.get_major_flavours()))
        print(f"| {i}    | {r.name[:32]:32} | {fitness:7.2f} | {base_count:9d} | {addin_count:14d} | {topping_count:8d} | {major_flavours:27} |")

    # Flavor diversity using average pairwise Jaccard distance
    def jaccard_distance(s1, s2):
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return 0 if union == 0 else 1 - intersection / union

    pairs = list(combinations(selected_recipes, 2))
    flavor_distances = [jaccard_distance(a.get_major_flavours(), b.get_major_flavours()) for a, b in pairs]
    avg_flavor_distance = np.mean(flavor_distances) if flavor_distances else 0

    # Ingredient overlap: average pairwise Jaccard similarity on all ingredients
    def ingredient_set(r):
        return set(clean_token(i['ingredient']) for i in r.all_ingredients() if clean_token(i['ingredient']))

    ingredient_overlaps = []
    for a, b in pairs:
        s1, s2 = ingredient_set(a), ingredient_set(b)
        similarity = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
        ingredient_overlaps.append(similarity)
    avg_ingredient_overlap = np.mean(ingredient_overlaps) if ingredient_overlaps else 0

    print(f"\nFlavor Diversity (Average Pairwise Jaccard Distance): {avg_flavor_distance:.3f}")
    print(f"Ingredient Overlap (Average Pairwise Jaccard Similarity): {avg_ingredient_overlap:.3f}\n")
    print("----------------------\n")


def run_cookie_generator():
    with open("recipes.json", "r", encoding="utf-8") as f:
        data = json.load(f)["recipes"]

    print("\nGenerating new cookie recipes...\n")
    ga = CookieGeneticAlgorithm(data, mutation_rate=0.6, population_size=20)
    population = ga.evolve_population(generations=8, max_attempts=200)
    distinct_recipes = ga.select_three_distinct(population, fitness_threshold=3)

    for i, r in enumerate(distinct_recipes, 1):
        print(f"{i}. {r.name}")
        for sec in ["base_ingredients", "flavour_addins", "external_toppings"]:
            section = getattr(r, sec)
            if section:
                print(f" {sec.replace('_', ' ').title()}:")
                for ing in section:
                    print(f"  - {ing['amount']} {ing['unit']} {ing['ingredient']}")
        print(f" Fitness Score: {ga.evaluate_fitness(r)}\n")

    print_report_metrics(population, distinct_recipes, ga)
    visualize_recipe_space(data, distinct_recipes)


if __name__ == "__main__":
    run_cookie_generator()
