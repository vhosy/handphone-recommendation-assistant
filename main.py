import logging

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from codes.tools import guardrail_tool, recommendation_tool, cross_sell_tool,\
    checkout_tool, get_cust_tool, jaibreak_tool, cross_sell_outcome_tool

app = FastAPI()

# Create the agent
memory = MemorySaver()
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
tools = [recommendation_tool, cross_sell_tool, checkout_tool, guardrail_tool,\
         get_cust_tool, jaibreak_tool,  cross_sell_outcome_tool]
prompt = """You are a friendly handphone recommender assistant for customer.  
At the start, customer would first need to input their customer ID so that 
you can retrieve the current phone they possess with [get_cust_tool].  
Certain criteria to look out for:
1) If customer does not want to provide customer ID, ask for their phone preferences 
    and go to 'next step'.
2) If customer say their phone preferences before providing customer ID, remind 
    customer to provide customer ID first, but just remind one time is enough.
    go to 'next step'.

'next step'
Provide recommendations based on customer preferences.
Do occasionally check with the customer if they want the phone model.
If customer shows interest in the phone model recommended, or if customer says 
he wants the phone the system specified or outrights mentions the phone model they want,
use [cross_sell_tool] first.
After [cross_sell_tool] done, use  [cross_sell_outcome_tool].
Finally use [checkout_tool].
"""

agent = create_react_agent(model, tools, checkpointer = memory, prompt = prompt)
# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

logging.basicConfig(
    filename=f'response_log/response_{config['configurable']['thread_id']}.txt', 
    level=logging.INFO,         # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
)
logger = logging.getLogger(__name__)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Store messages in memory (in production, use a DB or session)
chat_history = [
    {"sender": "assistant", 
     "message": """Hi! I am your friendly handset recommender assistant! Before we start, can you please provide the numeric portion of yout customer id?
     """}
]
logger.info(f'ai\t\t{chat_history[0]["message"]}')

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})

@app.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, user_message: str = Form(...)):
    chat_history.append({"sender": "user", "message": user_message})
    
    for step in agent.stream({"messages": [{"role": "user", "content": user_message}]}, config, stream_mode="values"):
        message = step["messages"][-1] 
      
        logger.info(f'{message.type}\t{message.name}\t{message.content}')
      
    chat_history.append({"sender": "assistant", "message": message.content})
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})