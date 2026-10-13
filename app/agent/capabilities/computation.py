"""Group G - computation. A restricted AST evaluator: no names beyond an allowlist, no attribute
access, no calls except whitelisted math functions, bounded size/exponent. Not Python eval."""
import ast
import math
import operator as op
import statistics

from pydantic import BaseModel, Field

from app.agent.contracts import CapabilityError, ErrorCode
from app.agent.registry import Capability

MAX_LEN, MAX_NODES, MAX_ABS = 400, 200, 1e15
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: op.pow}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs, "round": round,
          "min": min, "max": max, "sum": sum, "floor": math.floor, "ceil": math.ceil,
          "mean": statistics.fmean, "median": statistics.median, "stdev": statistics.stdev}
_CONST = {"pi": math.pi, "e": math.e}


class CalcIn(BaseModel):
    expression: str = Field(min_length=1, max_length=MAX_LEN)
    variables: dict[str, float | list[float]] = Field(default={}, max_length=20)


class CalcOut(BaseModel):
    expression: str
    result: float
    summary: str = ""


def _eval(node, env, count):
    count[0] += 1
    if count[0] > MAX_NODES:
        raise ValueError("expression too complex")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _CONST:
            return _CONST[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env, count) for e in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        l, r = _eval(node.left, env, count), _eval(node.right, env, count)
        if isinstance(node.op, ast.Pow) and (abs(r) > 64 or abs(l) > 1e6):
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand, env, count))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a, env, count) for a in node.args])
    raise ValueError("unsupported syntax")


def evaluate(expression: str, variables: dict | None = None) -> float:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval(tree.body, dict(variables or {}), [0])
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError, statistics.StatisticsError) as exc:
        raise CapabilityError(ErrorCode.invalid_input, f"Cannot evaluate expression: {exc}") from exc
    if isinstance(value, list) or not isinstance(value, (int, float)) or math.isnan(value) or abs(value) > MAX_ABS:
        raise CapabilityError(ErrorCode.invalid_input, "Result is not a finite number within range.")
    return float(value)


async def run_calculation(a: CalcIn, ctx):
    r = evaluate(a.expression, a.variables)
    return CalcOut(expression=a.expression, result=r, summary=f"= {r:g}")


CAPS = [Capability("run_calculation", "computation", "Deterministic arithmetic/statistics: + - * / // % **, sqrt, log, exp, "
                   "round, min, max, sum, mean, median, stdev, pi, e, with named variables. Use instead of mental math.",
                   CalcIn, CalcOut, run_calculation)]
