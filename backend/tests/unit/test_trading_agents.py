"""
Tests for TradingAgents subsystem.

Covers: ConditionalLogic, Propagator, SignalProcessor, Reflector,
        dataflows utilities, and config management.

Uses spec_from_file_location to bypass __init__.py import chains
that pull in uninstalled optional deps (stockstats, langchain_openai, etc.).
"""

import sys
import os
import types
import pytest
from unittest.mock import MagicMock
from datetime import date, datetime
from typing import Annotated
import pandas as pd
import importlib.util


# ---------------------------------------------------------------------------
# Stub external deps before any TradingAgents imports
# ---------------------------------------------------------------------------

def _ensure_stub(name, attrs=None):
    """Register a MagicMock module stub if the real module isn't installed."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


# TypedDict base that AgentState(MessagesState) can inherit from
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict


class _FakeMessagesState(TypedDict):
    """Minimal stub for langgraph.graph.MessagesState."""
    messages: list


# langchain_openai
_ensure_stub("langchain_openai", {"ChatOpenAI": MagicMock})

# langgraph hierarchy
_ensure_stub("langgraph")
_ensure_stub("langgraph.graph", {
    "MessagesState": _FakeMessagesState,
    "END": "END",
    "START": "START",
    "StateGraph": MagicMock,
})
_ensure_stub("langgraph.prebuilt", {"ToolNode": MagicMock})

# tradingagents.agents — stub the package so `from tradingagents.agents import *` is a no-op
_ensure_stub("tradingagents")
_ensure_stub("tradingagents.agents")
_ensure_stub("tradingagents.agents.utils")

# chromadb and openai (for memory.py if we ever test it)
_ensure_stub("chromadb")
_ensure_stub("chromadb.config", {"Settings": MagicMock})
_ensure_stub("openai", {"OpenAI": MagicMock})

# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

_TRADING_AGENTS_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "TradingAgents", "tradingagents",
))


def _load_file(relative_path: str, module_name: str):
    """Load a single .py file directly, bypassing package __init__.py chains."""
    full_path = os.path.join(_TRADING_AGENTS_ROOT, relative_path)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load agent_states so other modules can import it
_agent_states = _load_file(
    "agents/utils/agent_states.py",
    "tradingagents.agents.utils.agent_states",
)

# Pre-load default_config so config.py can import it
_default_config = _load_file("default_config.py", "tradingagents.default_config")


# ---------------------------------------------------------------------------
# dataflows/utils.py tests
# ---------------------------------------------------------------------------

class TestDataflowUtils:
    """Tests for tradingagents.dataflows.utils."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_file("dataflows/utils.py", "ta_dataflows_utils")

    def test_get_current_date_format(self):
        result = self.mod.get_current_date()
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"
        assert result == date.today().strftime("%Y-%m-%d")

    def test_get_next_weekday_weekday_string(self):
        result = self.mod.get_next_weekday("2025-01-15")
        assert isinstance(result, datetime)
        assert result.weekday() < 5
        assert result == datetime(2025, 1, 15)

    def test_get_next_weekday_saturday_string(self):
        result = self.mod.get_next_weekday("2025-01-18")
        assert result.weekday() == 0
        assert result == datetime(2025, 1, 20)

    def test_get_next_weekday_sunday_string(self):
        result = self.mod.get_next_weekday("2025-01-19")
        assert result.weekday() == 0
        assert result == datetime(2025, 1, 20)

    def test_get_next_weekday_datetime_input(self):
        sat = datetime(2025, 1, 18)
        result = self.mod.get_next_weekday(sat)
        assert result.weekday() == 0
        assert result == datetime(2025, 1, 20)

    def test_get_next_weekday_friday_stays(self):
        result = self.mod.get_next_weekday("2025-01-17")
        assert result == datetime(2025, 1, 17)

    def test_save_output_with_path(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = str(tmp_path / "out.csv")
        self.mod.save_output(df, "test data", path)
        loaded = pd.read_csv(path)
        assert list(loaded["a"]) == [1, 2]

    def test_save_output_no_path(self):
        df = pd.DataFrame({"x": [1]})
        self.mod.save_output(df, "no save", None)

    def test_decorate_all_methods(self):
        def add_prefix(func):
            def wrapper(*args, **kwargs):
                return "prefix_" + func(*args, **kwargs)
            return wrapper

        @self.mod.decorate_all_methods(add_prefix)
        class MyClass:
            def greet(self):
                return "hello"
            def farewell(self):
                return "bye"

        obj = MyClass()
        assert obj.greet() == "prefix_hello"
        assert obj.farewell() == "prefix_bye"


# ---------------------------------------------------------------------------
# dataflows/config.py tests
# ---------------------------------------------------------------------------

class TestDataflowConfig:
    """Tests for tradingagents.dataflows.config."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_file("dataflows/config.py", "ta_dataflows_config")
        self.mod._config = None
        self.mod.DATA_DIR = None
        self.mod.initialize_config()

    def test_get_config_returns_dict(self):
        cfg = self.mod.get_config()
        assert isinstance(cfg, dict)
        assert "data_dir" in cfg

    def test_get_config_returns_copy(self):
        cfg1 = self.mod.get_config()
        cfg2 = self.mod.get_config()
        assert cfg1 is not cfg2

    def test_set_config_updates(self):
        self.mod.set_config({"custom_key": "custom_value"})
        updated = self.mod.get_config()
        assert updated["custom_key"] == "custom_value"

    def test_set_config_preserves_existing(self):
        self.mod.set_config({"new_key": 42})
        updated = self.mod.get_config()
        assert "data_dir" in updated
        assert updated["new_key"] == 42

    def test_initialize_config_from_scratch(self):
        self.mod._config = None
        self.mod.DATA_DIR = None
        self.mod.initialize_config()
        assert self.mod._config is not None
        assert self.mod.DATA_DIR is not None


# ---------------------------------------------------------------------------
# ConditionalLogic tests
# ---------------------------------------------------------------------------

class TestConditionalLogic:
    """Tests for tradingagents.graph.conditional_logic.ConditionalLogic."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_file(
            "graph/conditional_logic.py",
            "ta_graph_conditional_logic",
        )
        self.ConditionalLogic = mod.ConditionalLogic

    def _make_message(self, has_tool_calls: bool):
        msg = MagicMock()
        msg.tool_calls = [{"name": "tool"}] if has_tool_calls else []
        return msg

    def _make_state(self, messages=None, invest_debate=None, risk_debate=None):
        return {
            "messages": messages or [],
            "investment_debate_state": invest_debate or {
                "count": 0, "current_response": "",
            },
            "risk_debate_state": risk_debate or {
                "count": 0, "latest_speaker": "",
            },
        }

    def test_market_with_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(True)])
        assert cl.should_continue_market(state) == "tools_market"

    def test_market_without_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(False)])
        assert cl.should_continue_market(state) == "Msg Clear Market"

    def test_social_with_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(True)])
        assert cl.should_continue_social(state) == "tools_social"

    def test_social_without_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(False)])
        assert cl.should_continue_social(state) == "Msg Clear Social"

    def test_news_with_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(True)])
        assert cl.should_continue_news(state) == "tools_news"

    def test_news_without_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(False)])
        assert cl.should_continue_news(state) == "Msg Clear News"

    def test_fundamentals_with_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(True)])
        assert cl.should_continue_fundamentals(state) == "tools_fundamentals"

    def test_fundamentals_without_tool_calls(self):
        cl = self.ConditionalLogic()
        state = self._make_state(messages=[self._make_message(False)])
        assert cl.should_continue_fundamentals(state) == "Msg Clear Fundamentals"

    def test_debate_ends_at_max_rounds(self):
        cl = self.ConditionalLogic(max_debate_rounds=2)
        state = self._make_state(invest_debate={"count": 4, "current_response": "Bull"})
        assert cl.should_continue_debate(state) == "Research Manager"

    def test_debate_routes_to_bear_after_bull(self):
        cl = self.ConditionalLogic(max_debate_rounds=3)
        state = self._make_state(invest_debate={"count": 1, "current_response": "Bull says buy"})
        assert cl.should_continue_debate(state) == "Bear Researcher"

    def test_debate_routes_to_bull_after_bear(self):
        cl = self.ConditionalLogic(max_debate_rounds=3)
        state = self._make_state(invest_debate={"count": 1, "current_response": "Bear says sell"})
        assert cl.should_continue_debate(state) == "Bull Researcher"

    def test_debate_default_rounds_1(self):
        cl = self.ConditionalLogic()
        state = self._make_state(invest_debate={"count": 2, "current_response": "Bull"})
        assert cl.should_continue_debate(state) == "Research Manager"

    def test_risk_ends_at_max_rounds(self):
        cl = self.ConditionalLogic(max_risk_discuss_rounds=2)
        state = self._make_state(risk_debate={"count": 6, "latest_speaker": "Risky"})
        assert cl.should_continue_risk_analysis(state) == "Risk Judge"

    def test_risk_routes_safe_after_risky(self):
        cl = self.ConditionalLogic(max_risk_discuss_rounds=3)
        state = self._make_state(risk_debate={"count": 1, "latest_speaker": "Risky Analyst"})
        assert cl.should_continue_risk_analysis(state) == "Safe Analyst"

    def test_risk_routes_neutral_after_safe(self):
        cl = self.ConditionalLogic(max_risk_discuss_rounds=3)
        state = self._make_state(risk_debate={"count": 1, "latest_speaker": "Safe Analyst"})
        assert cl.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_risk_routes_risky_by_default(self):
        cl = self.ConditionalLogic(max_risk_discuss_rounds=3)
        state = self._make_state(risk_debate={"count": 1, "latest_speaker": "Neutral Analyst"})
        assert cl.should_continue_risk_analysis(state) == "Risky Analyst"


# ---------------------------------------------------------------------------
# Propagator tests
# ---------------------------------------------------------------------------

class TestPropagator:
    """Tests for tradingagents.graph.propagation.Propagator."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_file("graph/propagation.py", "ta_graph_propagation")
        self.Propagator = mod.Propagator

    def test_create_initial_state_structure(self):
        prop = self.Propagator()
        state = prop.create_initial_state("AAPL", "2025-01-15")
        assert state["company_of_interest"] == "AAPL"
        assert state["trade_date"] == "2025-01-15"
        assert state["market_report"] == ""
        assert state["fundamentals_report"] == ""
        assert state["sentiment_report"] == ""
        assert state["news_report"] == ""
        assert "investment_debate_state" in state
        assert "risk_debate_state" in state
        assert len(state["messages"]) == 1

    def test_create_initial_state_debate_counts_zero(self):
        prop = self.Propagator()
        state = prop.create_initial_state("TSLA", "2025-02-01")
        assert state["investment_debate_state"]["count"] == 0
        assert state["risk_debate_state"]["count"] == 0

    def test_create_initial_state_risk_debate_fields(self):
        prop = self.Propagator()
        state = prop.create_initial_state("MSFT", "2025-03-01")
        rd = state["risk_debate_state"]
        assert rd["history"] == ""
        assert rd["current_risky_response"] == ""
        assert rd["current_safe_response"] == ""
        assert rd["current_neutral_response"] == ""

    def test_get_graph_args_defaults(self):
        prop = self.Propagator()
        args = prop.get_graph_args()
        assert args["stream_mode"] == "values"
        assert args["config"]["recursion_limit"] == 100

    def test_get_graph_args_custom_limit(self):
        prop = self.Propagator(max_recur_limit=50)
        args = prop.get_graph_args()
        assert args["config"]["recursion_limit"] == 50


# ---------------------------------------------------------------------------
# SignalProcessor tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestSignalProcessor:
    """Tests for tradingagents.graph.signal_processing.SignalProcessor."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_file(
            "graph/signal_processing.py",
            "ta_graph_signal_processing",
        )
        self.SignalProcessor = mod.SignalProcessor

    def _make_mock_llm(self, response_content: str):
        llm = MagicMock()
        result = MagicMock()
        result.content = response_content
        llm.invoke.return_value = result
        return llm

    def test_process_signal_buy(self):
        llm = self._make_mock_llm("BUY")
        sp = self.SignalProcessor(llm)
        decision = sp.process_signal("Analysts recommend aggressive buying position.")
        assert decision == "BUY"
        llm.invoke.assert_called_once()

    def test_process_signal_sell(self):
        llm = self._make_mock_llm("SELL")
        sp = self.SignalProcessor(llm)
        decision = sp.process_signal("Major downtrend expected. Sell immediately.")
        assert decision == "SELL"

    def test_process_signal_hold(self):
        llm = self._make_mock_llm("HOLD")
        sp = self.SignalProcessor(llm)
        decision = sp.process_signal("Market is uncertain. Hold positions.")
        assert decision == "HOLD"

    def test_process_signal_passes_correct_messages(self):
        llm = self._make_mock_llm("BUY")
        sp = self.SignalProcessor(llm)
        sp.process_signal("test signal")
        call_args = llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0][0] == "system"
        assert call_args[1][0] == "human"
        assert call_args[1][1] == "test signal"


# ---------------------------------------------------------------------------
# Reflector tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestReflector:
    """Tests for tradingagents.graph.reflection.Reflector."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_file("graph/reflection.py", "ta_graph_reflection")
        self.Reflector = mod.Reflector

    def _make_mock_llm(self, response: str = "reflection output"):
        llm = MagicMock()
        result = MagicMock()
        result.content = response
        llm.invoke.return_value = result
        return llm

    def _make_state(self):
        return {
            "market_report": "Market up 2%",
            "sentiment_report": "Positive sentiment",
            "news_report": "Fed holds rates",
            "fundamentals_report": "Strong earnings",
            "investment_debate_state": {
                "bull_history": "Bull argued growth",
                "bear_history": "Bear argued caution",
                "judge_decision": "BUY with moderate allocation",
            },
            "risk_debate_state": {
                "judge_decision": "Moderate risk acceptable",
            },
            "trader_investment_plan": "Buy 100 shares at market",
        }

    def test_reflector_init_sets_prompt(self):
        llm = self._make_mock_llm()
        r = self.Reflector(llm)
        assert "financial analyst" in r.reflection_system_prompt.lower()

    def test_extract_current_situation(self):
        llm = self._make_mock_llm()
        r = self.Reflector(llm)
        situation = r._extract_current_situation(self._make_state())
        assert "Market up 2%" in situation
        assert "Positive sentiment" in situation
        assert "Fed holds rates" in situation
        assert "Strong earnings" in situation

    def test_reflect_bull_researcher(self):
        llm = self._make_mock_llm("Bull reflection")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_bull_researcher(self._make_state(), "+5%", memory)
        llm.invoke.assert_called_once()
        memory.add_situations.assert_called_once()

    def test_reflect_bear_researcher(self):
        llm = self._make_mock_llm("Bear reflection")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_bear_researcher(self._make_state(), "-2%", memory)
        llm.invoke.assert_called_once()
        memory.add_situations.assert_called_once()

    def test_reflect_trader(self):
        llm = self._make_mock_llm("Trader reflection")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_trader(self._make_state(), "+3%", memory)
        llm.invoke.assert_called_once()
        memory.add_situations.assert_called_once()

    def test_reflect_invest_judge(self):
        llm = self._make_mock_llm("Judge reflection")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_invest_judge(self._make_state(), "+1%", memory)
        memory.add_situations.assert_called_once()

    def test_reflect_risk_manager(self):
        llm = self._make_mock_llm("Risk reflection")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_risk_manager(self._make_state(), "-0.5%", memory)
        memory.add_situations.assert_called_once()

    def test_reflect_passes_returns_in_message(self):
        llm = self._make_mock_llm("output")
        r = self.Reflector(llm)
        memory = MagicMock()
        r.reflect_trader(self._make_state(), "+10%", memory)
        call_args = llm.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "+10%" in human_msg


# ---------------------------------------------------------------------------
# AgentState TypedDict structure tests
# ---------------------------------------------------------------------------

class TestAgentStates:
    """Tests for tradingagents.agents.utils.agent_states TypedDict definitions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.InvestDebateState = _agent_states.InvestDebateState
        self.RiskDebateState = _agent_states.RiskDebateState
        self.AgentState = _agent_states.AgentState

    def test_invest_debate_state_keys(self):
        annotations = self.InvestDebateState.__annotations__
        expected = {"bull_history", "bear_history", "history",
                    "current_response", "judge_decision", "count"}
        assert expected == set(annotations.keys())

    def test_risk_debate_state_keys(self):
        annotations = self.RiskDebateState.__annotations__
        expected = {
            "risky_history", "safe_history", "neutral_history",
            "history", "latest_speaker",
            "current_risky_response", "current_safe_response",
            "current_neutral_response", "judge_decision", "count",
        }
        assert expected == set(annotations.keys())

    def test_agent_state_has_company_field(self):
        annotations = self.AgentState.__annotations__
        assert "company_of_interest" in annotations
        assert "trade_date" in annotations

    def test_agent_state_has_report_fields(self):
        annotations = self.AgentState.__annotations__
        for field in ["market_report", "sentiment_report",
                      "news_report", "fundamentals_report"]:
            assert field in annotations, f"Missing field: {field}"

    def test_agent_state_has_debate_fields(self):
        annotations = self.AgentState.__annotations__
        assert "investment_debate_state" in annotations
        assert "risk_debate_state" in annotations
        assert "investment_plan" in annotations
        assert "trader_investment_plan" in annotations
        assert "final_trade_decision" in annotations


# ---------------------------------------------------------------------------
# default_config tests
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    """Tests for tradingagents.default_config."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _default_config

    def test_default_config_exists(self):
        assert hasattr(self.mod, "DEFAULT_CONFIG")
        assert isinstance(self.mod.DEFAULT_CONFIG, dict)

    def test_default_config_has_llm_settings(self):
        cfg = self.mod.DEFAULT_CONFIG
        assert "llm_provider" in cfg
        assert "deep_think_llm" in cfg
        assert "quick_think_llm" in cfg
        assert "backend_url" in cfg

    def test_default_config_has_debate_settings(self):
        cfg = self.mod.DEFAULT_CONFIG
        assert "max_debate_rounds" in cfg
        assert "max_risk_discuss_rounds" in cfg
        assert "max_recur_limit" in cfg
        assert isinstance(cfg["max_debate_rounds"], int)
        assert isinstance(cfg["max_risk_discuss_rounds"], int)

    def test_default_config_has_dirs(self):
        cfg = self.mod.DEFAULT_CONFIG
        assert "project_dir" in cfg
        assert "results_dir" in cfg
        assert "data_dir" in cfg
        assert "data_cache_dir" in cfg

    def test_default_config_online_tools(self):
        cfg = self.mod.DEFAULT_CONFIG
        assert "online_tools" in cfg
        assert isinstance(cfg["online_tools"], bool)
