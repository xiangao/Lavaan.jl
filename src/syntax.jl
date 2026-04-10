# ─── syntax.jl ────────────────────────────────────────────────────────────────
# Parse a lavaan model string into a flat list of parameter-table rows.
#
# Supported operators (in detection priority order, longest first):
#   =~   measurement (factor → indicators)
#   <~   composite / formative
#   ~*~  scaling parameter (for categorical parameterization)
#   ~~   (co)variance
#   ~1   intercept (special: no rhs token, rhs fixed to "1")
#   ~    regression
#   |~   alias for ~~  (not used in practice, kept for compatibility)
#   |    threshold (ordinal)
#   :=   user parameter definition
#   ==   equality constraint
#   <    inequality constraint (less-than)
#   >    inequality constraint (greater-than)
#   %    group weight
#   :    block declaration (group: / level:)
#
# Each non-constraint formula `LHS op RHS1 + RHS2 + ...` is expanded into
# one row per (LHS, RHS) pair, each potentially carrying modifiers:
#   value * x       → fixed value (e.g. 1*x1)
#   start(v) * x   → starting value modifier
#   label("l") * x → user label
#   lower(v) * x   → lower bound
#   upper(v) * x   → upper bound
#   equal("l") * x → alias for label (shares free index with same label)
#
# Returns a Vector of NamedTuples with fields:
#   lhs, op, rhs, user_fixed, user_start, user_label, user_lower, user_upper
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParsedRow

One expanded row from the model syntax, before parTable construction.
"""
struct ParsedRow
    lhs::String
    op::String
    rhs::String
    # modifier fields (all optional)
    fixed::Union{Float64,Nothing}    # fixed value (e.g. 1*x or 0*x)
    start::Union{Float64,Nothing}    # start(v)
    label::String                    # user-defined parameter label
    lower::Float64                   # lower bound (-Inf if not set)
    upper::Float64                   # upper bound (+Inf if not set)
    block::Int                       # block index (from : operator)
    group::String                    # group label
    level::String                    # level label
    efa_block::String                # efa block name (for EFA syntax)
end

function ParsedRow(lhs, op, rhs;
                   fixed=nothing, start=nothing, label="",
                   lower=-Inf, upper=Inf,
                   block=1, group="", level="", efa_block="")
    ParsedRow(string(lhs), string(op), string(rhs),
              fixed, start, label, lower, upper, block, group, level, efa_block)
end

# ─── Main entry point ─────────────────────────────────────────────────────────

"""
    parse_model_string(model::String) → Vector{ParsedRow}

Parse a lavaan model string into a flat list of expanded parameter rows.

# Example
```julia
rows = parse_model_string(\"\"\"
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    visual ~~ textual
\"\"\")
```
"""
function parse_model_string(model::AbstractString)::Vector{ParsedRow}
    # Normalize: replace Unicode tilde variants, collapse line continuations
    src = replace(string(model), '\u02dc' => '~')

    # Strip inline comments (# to end of line)
    src = replace(src, r"#[^\n]*" => "")

    # Split into logical lines, handling continuation (+/~ at line end means continue)
    lines = _split_logical_lines(src)

    # Parse each logical line
    rows = ParsedRow[]
    current_block = 0
    current_group = ""
    current_level = ""

    for line in lines
        line = String(strip(line))
        isempty(line) && continue

        # Use max(1, current_block) for rows before any explicit block declaration
        parsed = _parse_line(line, max(1, current_block), current_group, current_level)
        if parsed === nothing
            continue
        elseif parsed isa Tuple{<:AbstractString, <:AbstractString, Int}
            # Block declaration: (type, label, block_idx)
            btype, blabel, _ = parsed
            if btype == "group"
                current_group = blabel
                current_level = ""
                current_block += 1
            elseif btype == "level"
                current_level = blabel
                current_block += 1
            end
        else
            if current_block == 0
                current_block = 1
            end
            append!(rows, parsed)
        end
    end

    return rows
end

# ─── Internal: split into logical lines ───────────────────────────────────────

"""
Split the raw source into logical lines.
A logical line continues if it ends in `+`, `~`, `=~`, etc. (continuation chars).
"""
function _split_logical_lines(src::String)::Vector{String}
    # Replace \r\n and \r with \n
    src = replace(src, "\r\n" => "\n", "\r" => "\n")
    raw_lines = split(src, '\n')

    logical = String[]
    buffer = ""
    for line in raw_lines
        stripped = rstrip(line)
        if !isempty(buffer)
            buffer = buffer * " " * lstrip(stripped)
        else
            buffer = stripped
        end
        # Continuation: line ends with + (more RHS terms coming)
        if endswith(rstrip(buffer), '+')
            continue  # accumulate
        end
        push!(logical, buffer)
        buffer = ""
    end
    !isempty(strip(buffer)) && push!(logical, buffer)
    return logical
end

# ─── Operator detection (order matters: longest first) ────────────────────────

const LAVAAN_OPS = ["=~", "<~", "~*~", "~~", ":=", "==", "~1", "~", "|~", "|", "%", "<", ">", ":"]

const CONSTRAINT_OPS = Set(["==", ":=", "<", ">"])
const BLOCK_OPS      = Set([":"])
const INTERCEPT_OP   = "~1"

"""
Find the main operator in a formula line.
Returns (op, lhs_part, rhs_part) or nothing.
"""
function _find_operator(line::AbstractString)
    # Try operators in priority order (longest first)
    for op in LAVAAN_OPS
        # Use word-boundary-aware search to avoid, e.g., matching ~ inside ~~
        idx = _find_op_position(line, op)
        idx === nothing && continue

        lhs = strip(line[1:idx-1])
        rhs = strip(line[idx+length(op):end])
        isempty(lhs) && continue

        # ~1 special: no rhs token
        if op == "~1"
            return (op, lhs, "1")
        end

        return (op, lhs, rhs)
    end
    return nothing
end

"""
Find the position of operator `op` in `line`, avoiding false matches
(e.g., don't match ~ when looking for ~~).
Returns the byte index of the start of the operator, or nothing.
"""
function _find_op_position(line::AbstractString, op::AbstractString)::Union{Int,Nothing}
    i = 1
    n = length(line)
    olen = length(op)
    while i <= n - olen + 1
        if line[i:i+olen-1] == op
            # Check that this isn't a prefix of a longer operator
            # (e.g., don't match ~ when the next char makes it ~~)
            after = i + olen
            if after <= n
                rest = line[after:min(after, n)]
                # If op ends with ~ and next char is ~ → not a match for single ~
                if op == "~" && !isempty(rest) && rest[1] == '~'
                    i += 1; continue
                end
                # If op == "=" and next char is ~ → not a match for :=
                if op == "<" && !isempty(rest) && rest[1] == '~'
                    i += 1; continue
                end
            end
            # Don't match inside a modifier function call (preceded by letter/digit)
            if i > 1 && (isletter(line[i-1]) || isdigit(line[i-1]))
                if op in ("<", ">")
                    i += 1; continue
                end
            end
            return i
        end
        i += 1
    end
    return nothing
end

# ─── Internal: parse one logical line ─────────────────────────────────────────

function _parse_line(line::AbstractString, block::Int, group::AbstractString, level::AbstractString)
    result = _find_operator(line)
    result === nothing && return nothing

    op, lhs_raw, rhs_raw = result

    # Block declaration: e.g. "group: control"
    if op == ":"
        lhs_raw = strip(lhs_raw)
        rhs_raw = strip(rhs_raw)
        return (lhs_raw, rhs_raw, block)
    end

    # Constraint / definition rows: store as single row, lhs/rhs = raw strings
    if op in CONSTRAINT_OPS
        return [ParsedRow(strip(lhs_raw), op, strip(rhs_raw);
                          block=block, group=group, level=level)]
    end

    # Regular formula: expand LHS × RHS
    lhs_terms = _split_terms(lhs_raw)
    rhs_terms = _split_terms(rhs_raw)

    rows = ParsedRow[]
    for lhs_term in lhs_terms
        lhs_var, lhs_mod = _parse_term(lhs_term)
        for rhs_term in rhs_terms
            rhs_var, rhs_mod = _parse_term(rhs_term)
            # Merge modifiers (rhs modifiers take priority in lavaan)
            mod = _merge_modifiers(lhs_mod, rhs_mod)

            # Handle special case: op=~1, rhs is always "1"
            actual_rhs = (op == "~1") ? "1" : rhs_var

            push!(rows, ParsedRow(
                lhs_var, op, actual_rhs;
                fixed     = mod.fixed,
                start     = mod.start,
                label     = mod.label,
                lower     = mod.lower,
                upper     = mod.upper,
                block     = block,
                group     = group,
                level     = level,
                efa_block = mod.efa_block,
            ))
        end
    end
    return rows
end

# ─── Internal: split "a + b + c" into ["a", "b", "c"] ────────────────────────

"""
Split an expression on top-level `+` signs (not inside parentheses).
"""
function _split_terms(expr::AbstractString)::Vector{String}
    terms = String[]
    depth = 0
    start = 1
    for (i, c) in enumerate(expr)
        if c == '('; depth += 1
        elseif c == ')'; depth -= 1
        elseif c == '+' && depth == 0
            t = String(strip(expr[start:i-1]))
            !isempty(t) && push!(terms, t)
            start = i + 1
        end
    end
    t = String(strip(expr[start:end]))
    !isempty(t) && push!(terms, t)
    return terms
end

# ─── Modifier struct ──────────────────────────────────────────────────────────

Base.@kwdef mutable struct Modifier
    fixed::Union{Float64,Nothing}  = nothing
    start::Union{Float64,Nothing}  = nothing
    label::String                  = ""
    lower::Float64                 = -Inf
    upper::Float64                 = +Inf
    efa_block::String              = ""
end

function _merge_modifiers(lhs::Modifier, rhs::Modifier)::Modifier
    Modifier(
        fixed     = rhs.fixed !== nothing ? rhs.fixed : lhs.fixed,
        start     = rhs.start !== nothing ? rhs.start : lhs.start,
        label     = !isempty(rhs.label) ? rhs.label : lhs.label,
        lower     = isfinite(rhs.lower) ? rhs.lower : lhs.lower,
        upper     = isfinite(rhs.upper) ? rhs.upper : lhs.upper,
        efa_block = !isempty(rhs.efa_block) ? rhs.efa_block : lhs.efa_block,
    )
end

# ─── Internal: parse one term and extract modifiers ───────────────────────────

"""
Parse a single term like `start(0.5)*x1` or `label("b1")*x1` or `1*x1` or `x1`.
Returns (variable_name, Modifier).
"""
function _parse_term(term::AbstractString)::Tuple{String,Modifier}
    term = String(strip(term))
    mod = Modifier()

    # Iteratively strip modifier prefixes (there can be multiple chained)
    changed = true
    while changed
        changed = false

        # Pattern: numeric_literal * identifier  (e.g. 1*x1 or 0.5*x1)
        m = match(r"^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\*\s*(.+)$", term)
        if m !== nothing
            mod.fixed = parse(Float64, m.captures[1])
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: start(v)*x
        m = match(r"^start\s*\(\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\)\s*\*\s*(.+)$"i, term)
        if m !== nothing
            mod.start = parse(Float64, m.captures[1])
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: label("lbl")*x or equal("lbl")*x
        m = match(r"^(?:label|equal)\s*\(\s*[\"']?([a-zA-Z_][a-zA-Z0-9_.]*)[\"']?\s*\)\s*\*\s*(.+)$"i, term)
        if m !== nothing
            mod.label = m.captures[1]
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: identifier*variable — shorthand label (e.g. `a*x` → label="a", var="x")
        # Must come after numeric check so `1*x` still sets fixed=1.0 not label="1"
        m = match(r"^([a-zA-Z_][a-zA-Z0-9_.]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_.]*)$", term)
        if m !== nothing
            mod.label = m.captures[1]
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: lower(v)*x
        m = match(r"^lower\s*\(\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\)\s*\*\s*(.+)$"i, term)
        if m !== nothing
            mod.lower = parse(Float64, m.captures[1])
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: upper(v)*x
        m = match(r"^upper\s*\(\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*\)\s*\*\s*(.+)$"i, term)
        if m !== nothing
            mod.upper = parse(Float64, m.captures[1])
            term = String(strip(m.captures[2]))
            changed = true; continue
        end

        # Pattern: efa("block_name")*x  (EFA block membership)
        m = match(r"^efa\s*\(\s*[\"']([^\"']+)[\"']\s*\)\s*\*\s*(.+)$"i, term)
        if m !== nothing
            mod.efa_block = m.captures[1]
            term = String(strip(m.captures[2]))
            changed = true; continue
        end
    end

    return (term, mod)
end

# ─── Utility: extract variable names from parTable rows ───────────────────────

"""
    ov_names_from_rows(rows, lv_names) → Vector{String}

Extract observed variable names (variables that appear in model but are not latent).
"""
function ov_names_from_rows(rows::Vector{ParsedRow}, lv_names::Vector{String})::Vector{String}
    seen = Set{String}()
    for r in rows
        r.op == ":" && continue
        r.op in ("==", ":=", "<", ">") && continue
        # LHS of =~ / <~ is latent
        if r.op in ("=~", "<~")
            r.rhs != "1" && push!(seen, r.rhs)
        elseif r.op == "~1"
            push!(seen, r.lhs)
        elseif r.op in ("~~", "|~")
            !(r.lhs in lv_names) && push!(seen, r.lhs)
            !(r.rhs in lv_names) && r.rhs != "1" && push!(seen, r.rhs)
        elseif r.op == "~"
            # lhs of ~ is endogenous; skip if it's a latent variable
            !(r.lhs in lv_names) && push!(seen, r.lhs)
            !(r.rhs in lv_names) && r.rhs != "1" && push!(seen, r.rhs)
        end
    end
    # Remove latent variable names
    filter!(v -> !(v in lv_names), collect(seen))
    return sort(collect(seen))
end

"""
    lv_names_from_rows(rows) → Vector{String}

Extract latent variable names (LHS of =~ or <~).
"""
function lv_names_from_rows(rows::Vector{ParsedRow})::Vector{String}
    seen = OrderedSet{String}()
    for r in rows
        r.op in ("=~", "<~") && push!(seen, r.lhs)
    end
    return collect(seen)
end

# Simple ordered set via insertion-order Dict
mutable struct OrderedSet{T}
    d::Dict{T,Nothing}
    order::Vector{T}
    OrderedSet{T}() where T = new{T}(Dict{T,Nothing}(), T[])
end
function Base.push!(s::OrderedSet{T}, x::T) where T
    if !haskey(s.d, x)
        s.d[x] = nothing
        push!(s.order, x)
    end
    return s
end
Base.collect(s::OrderedSet) = copy(s.order)
Base.in(x, s::OrderedSet) = haskey(s.d, x)
