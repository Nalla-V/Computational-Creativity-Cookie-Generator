# The Great Bitwise Bake Off

## Overview
This project implements a genetic algorithm system to generate novel cookie recipes. Starting from a curated JSON knowledge base of recipes, the system evolves new recipes through crossover and mutation operators. Each recipe is evaluated using a fitness function that balances structural validity, ingredient diversity, and semantic novelty. The system selects diverse, high-quality recipes for presentation in a generative AI-assisted cookbook.

## Repository Structure
- `recipes.json` — JSON knowledge base containing 17 original cookie recipes.
- `cookie_generator.py` — Contains the full implementation of recipe representation, genetic algorithm operators, fitness evaluation, population evolution, diversity selection, and visualization.

## Requirements
- Python 3.8+
- Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Running Experiments
1. Execute the main script to load recipes, evolve the population over generations, and output the top recipes with fitness scores and details: 
```bash
python cookie_generator.py
```
The script also generates and saves a PCA plot (`embedding_space.png`) visualizing semantic relationships between original and generated recipes.

## Results
The system produces novel, diverse cookie recipes that maintain baking logic and creativity. Quantitative metrics such as Jaccard distances and embedding visualizations assess quality and novelty, supporting computational creativity claims.

## Cookbook Generation and Presentation
Final recipes from the generator can be formatted into a cookbook, complemented by AI-generated images (e.g., via Gemini-AI) and design tools like Canva to enhance visual appeal and creativity.




