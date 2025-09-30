# Gemini-Inspired Azure OpenAI Chatbot

This project provides a Streamlit application that mimics the Google Gemini app interface while connecting to an Azure OpenAI chat deployment. It is designed to run locally or on Azure App Service.

## Features
- Gemini-style hero section, floating header, and prompt suggestion chips.
- Chat interface powered by Azure OpenAI with token streaming for fast responses.
- Sidebar guidance for configuring environment variables on Azure App Service.
- "New chat" control for resetting the conversation without leaving the page.

## Requirements
Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Configure your Azure OpenAI credentials as environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_CHAT_DEPLOYMENT="<your-deployment-name>"
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

## Local development
Run the Streamlit application:

```bash
streamlit run streamlit_app/app.py
```

## Azure App Service deployment
1. Create an App Service with a Python runtime (3.10 or later recommended).
2. Deploy this repository (e.g., via GitHub Actions, Azure DevOps, or ZIP deploy).
3. Set the environment variables from the **Configuration** blade in the Azure portal.
4. Configure the startup command:
   ```bash
   streamlit run streamlit_app/app.py --server.port $PORT --server.address 0.0.0.0
   ```

The application will be available at the default App Service URL once deployment completes.
