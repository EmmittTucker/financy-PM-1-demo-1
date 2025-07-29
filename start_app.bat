@echo off
echo Starting AI Financial Portfolio Advisor...
echo.
echo Make sure you have set your HF_TOKEN in .env or .streamlit/secrets.toml
echo.
python -m streamlit run streamlit_app.py
pause
