import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ============================================================
# STEP 1: LOAD RECIPE DATA
# ============================================================
def load_recipes(filepath='recipes.json'):
    """Load recipes from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['recipes']

# ============================================================
# STEP 2: RECIPE FINDER CLASS
# ============================================================
class RecipeFinderRAG:
    def __init__(self, recipes_path='recipes.json'):
        """Initialize Recipe Finder with RAG"""
        print("🍳 Loading Recipes...")
        self.recipes = load_recipes(recipes_path)
        
        # Extract recipe information
        self.recipe_names = [r['name'] for r in self.recipes]
        self.recipe_descriptions = [r['description'] for r in self.recipes]
        self.all_ingredients = self._extract_all_ingredients()
        
        print(f"✓ Loaded {len(self.recipes)} recipes")
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for descriptions
        print("📝 Creating embeddings for recipes...")
        self.recipe_embeddings = self.embedding_model.encode(
            self.recipe_descriptions,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        print("🗂️ Building FAISS vector database...")
        dimension = self.recipe_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.recipe_embeddings))
        
        # Initialize LLM for recipe generation
        print("🤖 Loading LLM (GPT-2)...")
        self.generator = pipeline("text-generation", model="gpt2")
        
        print("✅ Recipe Finder initialized successfully!\n")
    
    def _extract_all_ingredients(self):
        """Extract all unique ingredients from recipes"""
        all_ingredients = set()
        for recipe in self.recipes:
            for ingredient in recipe['ingredients']:
                all_ingredients.add(ingredient['item'].lower())
        return sorted(list(all_ingredients))
    
    def find_recipes_by_ingredients(self, ingredients, top_k=3):
        """Find recipes matching given ingredients"""
        if isinstance(ingredients, str):
            ingredients = ingredients.split(',')
        
        ingredients = [ing.strip().lower() for ing in ingredients]
        
        matching_recipes = []
        
        for recipe in self.recipes:
            recipe_ingredients = [ing['item'].lower() for ing in recipe['ingredients']]
            matches = sum(1 for ing in ingredients if any(ing in r_ing for r_ing in recipe_ingredients))
            
            if matches > 0:
                matching_recipes.append({
                    'recipe': recipe,
                    'matches': matches,
                    'missing': [ing for ing in ingredients if not any(ing in r_ing for r_ing in recipe_ingredients)]
                })
        
        # Sort by number of matches
        matching_recipes.sort(key=lambda x: x['matches'], reverse=True)
        
        return matching_recipes[:top_k]
    
    def find_recipes_by_query(self, query, top_k=3):
        """Find recipes using semantic search"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k=top_k)
        
        retrieved_recipes = []
        for idx in indices[0]:
            retrieved_recipes.append({
                'recipe': self.recipes[idx],
                'similarity': float(1 / (1 + distances[0][indices[0].tolist().index(idx)]))
            })
        
        return retrieved_recipes
    
    def generate_customized_recipe(self, recipe, modifications=None):
        """Generate customized recipe using LLM"""
        recipe_text = f"""
Recipe: {recipe['name']}
Cuisine: {recipe['cuisine']}
Difficulty: {recipe['difficulty']}
Time: {recipe['time_minutes']} minutes
Servings: {recipe['servings']}

Ingredients:
"""
        for ing in recipe['ingredients']:
            recipe_text += f"- {ing['amount']} {ing['unit']} {ing['item']}\n"
        
        recipe_text += "\nInstructions:\n"
        for i, instruction in enumerate(recipe['instructions'], 1):
            recipe_text += f"{i}. {instruction}\n"
        
        # Create prompt for LLM
        modifications_text = f" with {modifications}" if modifications else ""
        
        prompt = f"""Here's a recipe:

{recipe_text}

Please provide a shorter, simplified summary of this recipe{modifications_text}. 
Make it beginner-friendly with clear, concise steps.

Summary:"""
        
        # Generate customized version
        response = self.generator(prompt, max_length=200, num_return_sequences=1)
        
        return response[0]['generated_text']
    
    def get_recipe_details(self, recipe_id):
        """Get detailed information about a recipe"""
        for recipe in self.recipes:
            if recipe['id'] == recipe_id:
                return recipe
        return None
    
    def format_recipe(self, recipe):
        """Format recipe for display"""
        output = f"""
╔════════════════════════════════════════════════════════════╗
║ {recipe['name'].upper()}
╚════════════════════════════════════════════════════════════╝

📍 Cuisine: {recipe['cuisine']} | ⏱️ Time: {recipe['time_minutes']} min | 👥 Servings: {recipe['servings']}
📊 Difficulty: {recipe['difficulty']}

📝 Description:
{recipe['description']}

🥘 Ingredients:
"""
        for ing in recipe['ingredients']:
            output += f"  • {ing['amount']} {ing['unit']} {ing['item']}\n"
        
        output += f"\n👨‍🍳 Instructions:\n"
        for i, instruction in enumerate(recipe['instructions'], 1):
            output += f"  {i}. {instruction}\n"
        
        output += "\n" + "="*60 + "\n"
        return output
    
    def search_and_suggest(self, user_input):
        """Main search and suggestion function"""
        print(f"\n👤 You: {user_input}\n")
        
        # Step 1: Try ingredient-based search first
        print("🔍 Searching for recipes...")
        ingredient_matches = self.find_recipes_by_ingredients(user_input, top_k=2)
        
        if ingredient_matches:
            print(f"✓ Found {len(ingredient_matches)} recipes with your ingredients!\n")
            return ingredient_matches
        
        # Step 2: Fall back to semantic search
        print("📖 Searching by recipe description...\n")
        semantic_matches = self.find_recipes_by_query(user_input, top_k=3)
        
        if semantic_matches:
            return [{'recipe': r['recipe'], 'matches': 0} for r in semantic_matches]
        
        print("❌ No recipes found matching your criteria.\n")
        return []

# ============================================================
# STEP 3: MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("    🍳 RECIPE FINDER WITH RAG 🍳")
    print("="*60)
    
    # Initialize Recipe Finder
    finder = RecipeFinderRAG('recipes.json')
    
    # Example searches
    print("\n" + "="*60)
    print("DEMO: Recipe Finder Examples")
    print("="*60)
    
    # Example 1: Search by ingredients
    print("\n🔎 EXAMPLE 1: Search by ingredients")
    print("Query: 'chicken, garlic, cream'")
    results = finder.search_and_suggest("chicken, garlic, cream")
    
    if results:
        for result in results[:2]:
            print(finder.format_recipe(result['recipe']))
    
    # Example 2: Semantic search
    print("\n🔎 EXAMPLE 2: Semantic search")
    print("Query: 'I want a quick and healthy meal'")
    results = finder.find_recipes_by_query("quick and healthy meal", top_k=2)
    
    for result in results:
        print(finder.format_recipe(result['recipe']))
    
    # Example 3: Generate customized recipe
    print("\n🔎 EXAMPLE 3: Generate customized recipe")
    recipe = finder.get_recipe_details(5)  # Butter Chicken Curry
    print(f"Original Recipe: {recipe['name']}\n")
    
    print("📝 Generating simplified version...")
    customized = finder.generate_customized_recipe(recipe, "vegetarian modifications")
    print(f"Customized Recipe:\n{customized}\n")
    
    # Interactive mode
    print("\n" + "="*60)
    print("🎯 Interactive Mode - Type 'exit' to quit")
    print("="*60)
    print("\nYou can search by:")
    print("  • Ingredients: 'chicken, rice, garlic'")
    print("  • Cuisine: 'Italian', 'Mexican', 'Indian'")
    print("  • Type: 'dessert', 'salad', 'curry'")
    print("  • Preference: 'quick and easy', 'healthy'")
    print()
    
    while True:
        user_input = input("🔍 Search for recipe: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Thank you for using Recipe Finder!")
            break
        
        if not user_input:
            continue
        
        # Search and display results
        results = finder.search_and_suggest(user_input)
        
        if results:
            for i, result in enumerate(results[:2], 1):
                print(finder.format_recipe(result['recipe']))
                
                # Ask if user wants customized version
                customize = input(f"Would you like a customized version of {result['recipe']['name']}? (y/n): ").strip().lower()
                if customize == 'y':
                    modifications = input("Enter modifications (or press Enter for default): ").strip()
                    customized = finder.generate_customized_recipe(result['recipe'], modifications)
                    print(f"\n📝 Customized Recipe:\n{customized}\n")
        else:
            print("❌ No recipes found. Try different ingredients or search terms.\n")