# Recipe Finder with RAG

A powerful recipe discovery and customization system using Retrieval-Augmented Generation (RAG).

## Features

- 🔍 **Ingredient-Based Search**: Find recipes by ingredients you have
- 📖 **Semantic Search**: Search recipes by description, cuisine, or preference
- 🤖 **LLM-Powered Customization**: Generate customized recipe versions
- ⚡ **Fast Retrieval**: Uses FAISS for instant recipe lookup
- 💾 **8 Diverse Recipes**: Italian, Chinese, Indian, Mexican, and more

## Installation

```bash
pip install -r requirements.txt

```

## Usage

### Run Interactive Mode

```bash
python src/recipe_finder.py
```

### Search Examples

```
🔍 Search by ingredients:
   "chicken, garlic, cream"

🔍 Search by cuisine:
   "Italian pasta"

🔍 Search by type:
   "quick and easy"
   "healthy salad"
   "spicy curry"
```

## Project Structure

```
Recipe_Finder_RAG/
├── data/
│   └── recipes.json           # Recipe database
├── src/
│   └── recipe_finder.py       # Main application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## How It Works

### 1. **Ingredient Search**
   - Analyzes ingredients you have
   - Matches with recipe ingredients
   - Ranks by number of matches

### 2. **Semantic Search**
   - Converts recipes to embeddings
   - Finds similar recipes by meaning
   - Uses FAISS for fast retrieval

### 3. **Recipe Customization**
   - Takes original recipe
   - Uses LLM to generate customized version
   - Can add dietary restrictions or preferences

## Technologies

- **sentence-transformers**: Recipe embeddings
- **FAISS**: Vector similarity search
- **transformers**: LLM for customization
- **PyTorch**: Deep learning backend

## Example Queries

```python
# Search by ingredients
finder.search_and_suggest("chicken, rice, garlic")

# Search by description
finder.find_recipes_by_query("healthy and quick")

# Get customized recipe
customized = finder.generate_customized_recipe(recipe, "vegetarian")
```

## Recipes Included

1. 🍝 Spaghetti Carbonara (Italian)
2. 🥦 Vegetable Stir Fry (Chinese)
3. 🍗 Grilled Chicken with Lemon (Mediterranean)
4. 🍪 Chocolate Chip Cookies (American)
5. 🍛 Butter Chicken Curry (Indian)
6. 🥗 Caesar Salad (Italian-American)
7. 🌮 Tacos al Pastor (Mexican)
8. 🐟 Salmon with Vegetables (Scandinavian)

## Future Enhancements

- Add nutritional information
- Dietary restriction filters (vegan, gluten-free)
- User ratings and reviews
- Shopping list generation
- Meal planning
- Recipe scaling for different servings

## License

MIT

## Author

M.laxmana chary
