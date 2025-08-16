.PHONY: train run predict

train:
\tpython -m training.train

run:
\tuvicorn serving_app.main:app --host 0.0.0.0 --port 8011 --reload

predict:
\tcurl -s -X POST http://localhost:8011/predict -H 'Content-Type: application/json' -d '{"features":[5.1,3.5,1.4,0.2], "return_proba":true}' | python -m json.tool
