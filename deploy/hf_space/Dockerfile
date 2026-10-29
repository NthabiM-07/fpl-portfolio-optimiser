# Hugging Face Docker Space for the Streamlit demo (Streamlit is no longer a built-in Spaces SDK;
# the Hub now creates Streamlit apps from the Docker template). Spaces run the container as user 1000.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user reports ./reports

EXPOSE 8501
CMD ["streamlit", "run", "src/fplopt/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
