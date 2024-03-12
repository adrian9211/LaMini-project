# PDF summarizer

# Usage
To run locally: streamlit run app.py

To host: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app

## Setup

1. Clone the repository to your local machine:

    ```bash
    git clone  https://github.com/adrian9211/LaMini-project
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3. Install all required pip packages:

    ```
    pip install -r requirements.txt
    ```

4. Install essential data model:

    ```
    git clone https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M
    ```

    or download here:

    https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M

## Running the App

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```
