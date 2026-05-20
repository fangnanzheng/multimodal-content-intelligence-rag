.PHONY: build test api dashboard

build:
	python scripts/build_demo.py --rows 1200

test:
	pytest -q

api:
	uvicorn content_intel.api.main:app --app-dir src --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app/streamlit_app.py --server.port 8502

