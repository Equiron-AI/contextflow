import locale
import pytz
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def get_time_string(tz='Europe/Moscow'):
    locale.setlocale(locale.LC_ALL, ('ru_RU', 'UTF-8'))
    dt = datetime.now(pytz.timezone(tz))
    return dt.strftime(" ### Текущее время: %A, %d %B, %Yг. %H:%M")


def parseToolResponse(response, tool_names):
    regex_pattern = r"(" + "|".join(item + ":" for item in tool_names) + ")"
    tools = re.split(regex_pattern, response)
    tools = [tool.strip() for tool in tools if tool.strip()]
    tools_dict = {}
    for i in range(1, len(tools), 2):
        tool_name = tools[i-1].strip(':').strip()
        tool_argument = tools[i].strip()
        tools_dict[tool_name] = tool_argument
        logger.info(tool_name + ": " + tool_argument)
    return tools_dict