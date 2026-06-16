# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""V2 tools — LangChain @tool functions for create_react_agent"""
from .web_search import web_search
from .bash import bash
from .python_exec import python_exec
from .plot import plot
from .read_file import read_file
from .write_file import write_file
from .edit import edit
from .grep import grep
from .create_plan import create_plan
from .workspace_ls import workspace_ls
from .browse_url import browse_url
from .browser_use_tool import browser_use
from .recall_web import recall_web
from .remember import remember
from .fetch_sitemap import fetch_sitemap
from .batch_browse import batch_browse
from .scrape_table import scrape_table
from .read_image import read_image
from .scratchpad import scratch_write, scratch_read, scratch_clear
from .tool_loop import tool_loop
from .skill_tools.research_orchestrator import research_orchestrator

ALL_TOOLS = [
    web_search, bash, python_exec, plot, read_file, write_file, edit, grep,
    create_plan, workspace_ls, browse_url, browser_use, recall_web, remember,
    fetch_sitemap, batch_browse, scrape_table,
    read_image,
    scratch_write, scratch_read, scratch_clear,
    tool_loop,
]

# Tools bound only when their matching skill mode is active.
# Add a new entry here for any future skill-only tool — no other code change needed.
SKILL_TOOLS: dict[str, list] = {
    "research": [research_orchestrator],
}
