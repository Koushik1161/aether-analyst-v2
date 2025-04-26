# main.py
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import traceback # Import traceback for detailed error logging

# Load .env file from the root directory BEFORE importing agent logic
load_dotenv()

# Import the LangGraph runner function
# Add error handling in case agent_graph fails during its own import (e.g., missing keys)
try:
    from orchestrator.agent_graph import run_agent_graph
except (ValueError, KeyError, ImportError) as e:
    # Set page config early for error message consistency
    st.set_page_config(page_title="Error - Aether Analyst", layout="centered", initial_sidebar_state="collapsed")
    st.error(f"❌ **Fatal Error:** Failed to initialize Agent Orchestrator: {e}")
    st.error("Please ensure all required API Keys (OpenAI, Grok, SerpAPI, NewsAPI, MCP_API_KEY) are correctly set in your `.env` file and all dependencies from `requirements.txt` are installed in your virtual environment.")
    st.code(traceback.format_exc())
    st.stop() # Stop the Streamlit app if agent fails to load
except Exception as e: # Catch any other unexpected import errors
    st.set_page_config(page_title="Error - Aether Analyst", layout="centered", initial_sidebar_state="collapsed")
    st.error(f"❌ **Fatal Error:** An unexpected error occurred during orchestrator import: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="✨ Aether Analyst",
    layout="centered", # Use centered layout for focused interaction
    initial_sidebar_state="collapsed",
    page_icon="✨"
)

# --- Custom CSS for Polished Dark Theme ---
st.markdown(
    """
    <style>
        /* Base font and background */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            background-color: #0F172A; /* Dark Slate Blue Background */
            color: #E2E8F0; /* Light Gray Text */
        }

        /* Main container adjustments */
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 800px;
            margin: auto;
        }

        /* Title styling */
        h1 {
            color: #F8FAFC; /* Near White Title */
            text-align: center;
            padding-bottom: 0.5rem;
            font-weight: 600;
            font-size: 2.5em;
        }
        /* Subtitle/description styling */
        .main .block-container > div:nth-child(1) > div > div > div > div:nth-child(2) > div > p {
            color: #94A3B8; /* Lighter Gray Text for description */
            text-align: center;
            margin-bottom: 2.5rem;
            font-size: 1.1em;
            line-height: 1.6;
        }

        /* Input area styling */
        .stTextInput label {
            color: #CBD5E1; /* Lighter Gray label */
            font-weight: 500;
            padding-bottom: 0.6rem;
            font-size: 1em;
        }
         .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #334155; /* Darker border */
            padding: 14px 16px;
            background-color: #1E293B; /* Darker input background */
            color: #F8FAFC; /* Light input text */
            font-size: 1em;
            transition: border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
         .stTextInput > div > div > input:focus {
             border-color: #60A5FA; /* Lighter blue focus */
             box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.3);
             outline: none;
             background-color: #293548; /* Slightly lighter focus background */
         }

        /* Button styling */
        .stButton>button {
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1em;
            border: none;
            background-color: #F8FAFC; /* Light button */
            color: #1E293B; /* Dark text on button */
            width: 100%;
            margin-top: 1rem;
            transition: background-color 0.2s ease-in-out, transform 0.1s ease;
        }
        .stButton>button:hover {
            background-color: #E2E8F0; /* Slightly darker light on hover */
            transform: translateY(-1px);
        }
        .stButton>button:active {
            background-color: #CBD5E1;
            transform: translateY(0px);
        }
        .stButton>button:focus {
             outline: none;
             box-shadow: 0 0 0 3px rgba(248, 250, 252, 0.3); /* Light focus ring */
        }

        /* Separators */
        hr {
            border-top: 1px solid #334155; /* Darker separator */
            margin-top: 2rem;
            margin-bottom: 2rem;
        }

        /* Results area styling */
        .stSpinner > div > div {
            text-align: center;
            padding: 2rem 0;
            color: #94A3B8; /* Lighter gray spinner text */
            font-size: 0.95em;
        }
        .results-section h3 {
             color: #F1F5F9; /* Lighter subheader */
             font-weight: 600;
             margin-top: 1rem;
             margin-bottom: 1.2rem;
             font-size: 1.4em;
             border-bottom: 1px solid #334155; /* Darker border */
             padding-bottom: 0.5rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid #334155;
            border-radius: 10px;
            background-color: #1E293B; /* Darker expander background */
            margin-top: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Softer shadow on dark */
        }
        [data-testid="stExpander"] summary {
            font-weight: 500;
            color: #94A3B8; /* Lighter gray text */
            font-size: 0.9em;
            padding: 8px 0;
        }
         [data-testid="stExpander"] summary:hover {
              color: #E2E8F0; /* Lighter hover text */
         }
         .stExpander > div { /* Content within expander */
             font-size: 0.9em;
             background-color: #0F172A; /* Match body background */
             border-top: 1px solid #334155;
         }
         .stExpander [data-testid="stJson"] { /* Style JSON within expander */
             background-color: #1E293B !important; /* Ensure dark background */
             border-radius: 8px;
         }


        /* Custom class for report markdown display */
        .report-markdown {
            background-color: #1E293B; /* Dark background for report */
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #334155; /* Darker border */
            color: #CBD5E1; /* Light gray text */
            line-height: 1.7;
            font-size: 1em;
        }
        .report-markdown h1, .report-markdown h2, .report-markdown h3, .report-markdown h4 {
            color: #F8FAFC; /* Near white headers */
            border-bottom: 1px solid #475569; /* Medium gray border */
            padding-bottom: 0.4em;
            margin-top: 1.8em;
            margin-bottom: 1em;
            font-weight: 600;
        }
         .report-markdown h1 { font-size: 1.6em; }
         .report-markdown h2 { font-size: 1.4em; }
         .report-markdown h3 { font-size: 1.2em; }
         .report-markdown h4 { font-size: 1.05em; border-bottom: none; font-weight: 600; }

        .report-markdown code {
            background-color: #0F172A; /* Match body background */
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-size: 85%;
            color: #94A3B8; /* Lighter gray for inline code */
            border: 1px solid #334155;
        }
        .report-markdown pre {
             background-color: #0F172A; /* Match body background */
             border-radius: 8px;
             padding: 15px;
             overflow-x: auto;
             border: 1px solid #334155;
             font-size: 0.9em;
        }
         .report-markdown pre code {
              background-color: transparent;
              padding: 0;
              border-radius: 0;
              border: none;
              color: #CBD5E1; /* Match surrounding text color */
              font-size: 1em;
         }
        .report-markdown a {
             color: #60A5FA; /* Lighter blue links */
             text-decoration: none;
             font-weight: 500;
        }
         .report-markdown a:hover {
             text-decoration: underline;
         }
         .report-markdown ul, .report-markdown ol {
              padding-left: 25px;
         }
          .report-markdown li {
              margin-bottom: 0.6em;
         }
         .report-markdown blockquote {
              border-left: 3px solid #475569; /* Medium gray quote border */
              padding-left: 1rem;
              margin-left: 0;
              color: #94A3B8; /* Lighter gray quote text */
              font-style: italic;
         }

        /* Footer info styling */
        .stAlert {
           background-color: transparent;
           border: none;
           color: #64748B; /* Darker subtle gray footer */
           border-radius: 8px;
           text-align: center;
           margin-top: 4rem;
           font-size: 0.85em;
           padding: 0.5rem;
        }

    </style>
    """, unsafe_allow_html=True
)

# --- App Layout ---
st.title("✨ Aether Analyst")
st.markdown("Your AI-powered Web Research Assistant")

# --- Input Section ---
with st.container():
    query = st.text_input(
        "Enter your research question",
        placeholder="e.g., Compare the latest AI safety regulations in the EU and USA.",
        key="query_input",
        label_visibility="collapsed"
    )
    submit_button = st.button("🚀 Run Research", key="run_button")


# --- Results Section ---
results_container = st.container()
results_container.markdown("---") # Separator before results area

if submit_button:
    if not query:
        results_container.error("⚠️ Please enter a question!")
    else:
        with results_container:
            with st.spinner("🧠 Aether Analyst is researching... This involves multiple steps and may take some time..."):
                try:
                    print(f"Starting agent run for query: {query}")
                    result_dict = asyncio.run(run_agent_graph(query))
                    print(f"Agent run finished.")

                    # Debug print (keep for now)
                    print(f"\nDEBUG (main.py): Result received from run_agent_graph:\n{result_dict}\n")

                    # Add class for specific styling
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)

                    st.subheader("Research Report") # Changed icon back to default text

                    markdown_content = result_dict.get("markdown") if isinstance(result_dict, dict) else None
                    is_error_message = isinstance(markdown_content, str) and ("Error:" in markdown_content or "Agent finished" in markdown_content) # More robust error check

                    if markdown_content and not is_error_message:
                        st.markdown(f'<div class="report-markdown">{markdown_content}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Agent run completed, but no final report content was generated or an error occurred during synthesis.")
                        if markdown_content and is_error_message:
                            st.error(f"Agent Error Message: {markdown_content}")
                        elif result_dict:
                             st.info("Raw agent output dictionary:")
                             st.json(result_dict)
                        else:
                             st.info("Result dictionary was None or Empty")

                    # Optional: Always display the final state data in an expander
                    with st.expander("Show Final Agent State (for debugging)"):
                        if result_dict and isinstance(result_dict, dict) and "data" in result_dict:
                            st.json(result_dict["data"], expanded=False)
                        elif result_dict and isinstance(result_dict, dict):
                             st.info("Final state data structure might be missing the 'data' key.")
                             st.json(result_dict, expanded=False)
                        else:
                            st.info("No detailed state data available.")

                    st.markdown('</div>', unsafe_allow_html=True) # Close results-section div

                except Exception as e:
                    st.error(f"🚨 An unexpected error occurred during the research process:")
                    st.exception(e)

# --- Footer ---
st.info("Aether Analyst uses multiple AI models and tools. Results are generated based on information retrieved from the web and may require verification.")