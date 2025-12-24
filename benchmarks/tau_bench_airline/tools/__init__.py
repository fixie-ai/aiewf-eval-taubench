"""TAU-bench airline tools dynamically loaded from original source.

This module loads tool definitions directly from the Tool classes
using the Tool.get_info() method to ensure exact alignment with the
original benchmark.
"""
from typing import List, Dict, Any
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from .book_reservation import BookReservation
from .calculate import Calculate
from .cancel_reservation import CancelReservation
from .get_reservation_details import GetReservationDetails
from .get_user_details import GetUserDetails
from .list_all_airports import ListAllAirports
from .search_direct_flight import SearchDirectFlight
from .search_onestop_flight import SearchOnestopFlight
from .send_certificate import SendCertificate
from .think import Think
from .transfer_to_human_agents import TransferToHumanAgents
from .update_reservation_baggages import UpdateReservationBaggages
from .update_reservation_flights import UpdateReservationFlights
from .update_reservation_passengers import UpdateReservationPassengers

# Mapping of tool names to their original classes for invocation
TOOL_CLASSES = {
    "book_reservation": BookReservation,
    "calculate": Calculate,
    "cancel_reservation": CancelReservation,
    "get_reservation_details": GetReservationDetails,
    "get_user_details": GetUserDetails,
    "list_all_airports": ListAllAirports,
    "search_direct_flight": SearchDirectFlight,
    "search_onestop_flight": SearchOnestopFlight,
    "send_certificate": SendCertificate,
    "think": Think,
    "transfer_to_human_agents": TransferToHumanAgents,
    "update_reservation_baggages": UpdateReservationBaggages,
    "update_reservation_flights": UpdateReservationFlights,
    "update_reservation_passengers": UpdateReservationPassengers,
}

def get_pipecat_tools() -> List[FunctionSchema]:
    """Extract FunctionSchema objects from original tool classes."""
    tools = []
    for name, cls in TOOL_CLASSES.items():
        info = cls.get_info()
        # info is in OpenAI format: {'type': 'function', 'function': {...}}
        func_info = info["function"]
        
        tools.append(FunctionSchema(
            name=func_info["name"],
            description=func_info["description"],
            properties=func_info["parameters"]["properties"],
            required=func_info["parameters"].get("required", [])
        ))
    return tools

# Create the ToolsSchema with all airline tools
AirlineToolsSchema = ToolsSchema(
    standard_tools=get_pipecat_tools()
)

__all__ = ["AirlineToolsSchema", "TOOL_CLASSES"]
