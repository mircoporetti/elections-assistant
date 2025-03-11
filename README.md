## Elections Programs AI

### Run the app locally

Requirements:

- Python 3.12
- Poetry

1. Create a `.env` file copied from `.env.example` and fill the variables.


2. Run

    ```bash
    make run
    ```
    or if you prefer the most annoying way:

    ```bash
    poetry install
    OPENAI_API_KEY=your_token API_USERNAME=elections-assistant API_PASSWORD=an_api_password poetry run uvicorn src.webapp.main:app --reload
    ```

### Run the tests with

```bash
make test
```