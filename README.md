# Fake Online Transaction Detection Using Random Forest

This Django web application provides a transaction risk assessment workflow backed by a Random Forest integration. No dataset or trained model is included yet, so the application reports that predictions are unavailable until you train a model on the selected dataset and save it to `models/random_forest_model.pkl`. The saved model is loaded once and cached; it is not retrained during requests.

## Project information
- Student: Vincent Kondowe
- Registration Number: 23311351014
- Project Serial Number: 7

## Technology stack
- Python
- Django
- scikit-learn
- pandas
- NumPy
- matplotlib
- SQLite

## Run locally
1. Create and activate a virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
4. Open http://127.0.0.1:8000/

## Project structure
- `detector/` contains the Django application and ML integration layer.
- `models/` is the destination for the trained Random Forest pickle file.
- `dataset/` keeps raw or processed transaction datasets.
- `notebooks/` is reserved for experimentation and model analysis.
- `results/` stores evaluation summaries and visual outputs.

The current transaction fields are provisional interface inputs. Replace them with the selected dataset's actual features before training and deployment. Do not report evaluation metrics until they have been measured on a held-out test set.
