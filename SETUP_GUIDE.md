# Setup Guide for CRM/ERP Connections

## Issues Fixed

1. ✅ **Fixed `NameError: name 'retrieved_context' is not defined`** - Added fallback context when embedding generation fails
2. ✅ **Fixed Ollama embeddings error** - Replaced non-existent `ollama embeddings` command with hash-based fallback
3. ✅ **Added actual CRM/ERP connections** - Integrated HubSpot and Odoo connectors into the Streamlit app

## Current Status

Your web app now has the infrastructure to connect to:
- **HubSpot CRM** (for deals, contacts, companies)
- **Odoo ERP** (for sales orders, invoices, partners)
- **Local Documents** (for your existing docs)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in your project root with your credentials:

```bash
# Copy the example file
cp config.example.env .env
```

Then edit `.env` with your actual credentials:

#### For HubSpot CRM:
1. Go to [HubSpot Developers](https://developers.hubspot.com/docs/api/private-apps)
2. Create a private app and get your access token
3. Add to `.env`:
```
HUBSPOT_ACCESS_TOKEN=your_actual_token_here
```

#### For Odoo ERP:
1. Get your Odoo instance URL, database name, username, and password
2. Add to `.env`:
```
ODOO_URL=https://your-odoo-instance.com
ODOO_DB=your_database_name
ODOO_USERNAME=your_username
ODOO_PASSWORD=your_password
```

### 3. Test Your Connections

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

You should see:
- ✅ Success messages for connected systems
- ⚠️ Warning messages for failed connections
- ❌ Error messages for configuration issues

### 4. Test Your Query

Try your original question: **"What are the common objections raised by prospects?"**

1. Select "CRM" as the data source
2. Enter your question
3. Click "Get Answer"

## Troubleshooting

### If you see "No data sources available":
- Check that your `.env` file exists and has the correct credentials
- Verify that your HubSpot/Odoo credentials are valid
- Check the console for specific error messages

### If you see "Failed to connect to HubSpot CRM":
- Verify your `HUBSPOT_ACCESS_TOKEN` is correct
- Check that your HubSpot account has the necessary permissions
- Ensure your private app has the required scopes

### If you see "Failed to connect to Odoo ERP":
- Verify your Odoo URL, database, username, and password
- Check that your Odoo instance is accessible
- Ensure your user has the necessary permissions

## Next Steps

1. **Test with real data**: Once connected, try different types of questions
2. **Customize prompts**: Modify the prompt templates in `src/rag/prompts/templates/`
3. **Add more data sources**: Extend the connectors for other systems
4. **Improve embeddings**: Replace the hash-based fallback with a proper embedding model

## Current Limitations

- **Embeddings**: Currently using a hash-based fallback. For production, consider using a proper embedding model like `sentence-transformers`
- **Rate limits**: Be aware of API rate limits for HubSpot and Odoo
- **Data scope**: Currently limited to basic objects (deals, contacts, companies for HubSpot; sales orders, invoices, partners for Odoo)
