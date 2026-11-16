-- Figures and tables: replace each float by a "**Figura N.** caption" / "**Tabla N.** caption" line.
-- The number comes from the .aux caption lines (captions.lua); when the .aux has no data for that kind,
-- from the label table, and as a last resort from a running count.
local S = require("state")
local text = require("strutil")
local labels = require("labels")
local captions = require("captions")
local M = {}

local function has_caption(c)
  return #(c.long or {}) > 0
end

-- Pick the number of a captioned float and keep the fallback counter in step.
-- `kind` ("figure"|"table") says which .aux entries exist; `counter` is S.fig_count or S.tab_count
-- (a table turned into a figure by the preprocessor still counts as a figure).
local function resolve_number(counter, kind, num, captioned)
  if num then
    local n = tonumber(num)
    if n then S[counter] = n end
    return num
  elseif captioned and not captions.has_aux(kind) then
    S[counter] = S[counter] + 1
    return tostring(S[counter])
  end
  return nil
end

local function lead_line(word, num, cap)
  local lead = pandoc.Inlines({ pandoc.Strong(pandoc.Str(word .. (num and (" " .. num) or "") .. ".")) })
  if #cap > 0 then lead:insert(pandoc.Space()); lead:extend(cap) end
  return lead
end

function M.Figure(fig)
  S.stats.figures = S.stats.figures + 1
  local num = labels.float_number(fig.identifier, "figure")
  local tables = {}
  fig.content:walk({ Table = function(t) tables[#tables + 1] = t end })
  local lab = fig.identifier ~= "" and S.labels[fig.identifier] or nil
  -- an image grid laid out with tabular is still a figure when its label says so
  if #tables > 0 and not (lab and lab.type == "figure") then
    -- A table environment wrapped as a figure: hand its caption to the first table and let the
    -- Table pass print it (as "Figura N" when the .aux lists it among the figures).
    S.stats.figures = S.stats.figures - 1
    local first = tables[1]
    if #(first.caption.long or {}) == 0 then first.caption = fig.caption end
    if first.identifier == "" then first.identifier = fig.identifier end
    local fnum = has_caption(first.caption)
      and captions.aux_number("figure", pandoc.utils.stringify(first.caption.long)) or nil
    if fnum then
      first.attr = pandoc.Attr(first.attr.identifier, first.attr.classes, { tex2md_fignum = fnum })
    end
    return tables
  end

  local captioned = has_caption(fig.caption)
  local word, kind = "Figura", "figure"
  -- a table without tabular was turned into a figure by the preprocessor and marked
  local marked = captioned and pandoc.utils.stringify(fig.caption.long):sub(1, 11) == "TEX2MDTABLE"
  if marked then word, kind = "Tabla", "table"; num = nil end
  if captioned then
    num = captions.aux_number(kind, (pandoc.utils.stringify(fig.caption.long):gsub("^TEX2MDTABLE", ""))) or num
  else
    num = nil  -- a panel without its own caption carries its parent's label: no number
  end
  num = resolve_number("fig_count", kind, num, captioned)
  local cap = text.strip_label_spans(text.caption_inlines(fig.caption))
  if marked then
    cap:remove(1)
    if cap[1] and cap[1].t == "Space" then cap:remove(1) end
  end
  return pandoc.Para(lead_line(word, num, cap))
end

function M.Table(t)
  S.stats.tables = S.stats.tables + 1
  local num = labels.float_number(t.identifier, "table")
  if not num then
    -- pandoc puts the \label of a table on a span inside the caption
    pandoc.walk_block(pandoc.Div(t.caption.long or {}), {
      Span = function(s) if s.identifier ~= "" then num = num or labels.float_number(s.identifier, "table") end end,
    })
  end
  local captioned = has_caption(t.caption)
  local word = "Tabla"
  if captioned then
    local fnum = t.attr.attributes["tex2md_fignum"]
    if fnum then
      num, word = fnum, "Figura"
    else
      num = captions.aux_number("table", pandoc.utils.stringify(t.caption.long)) or num
    end
  else
    num = nil
  end
  num = resolve_number("tab_count", "table", num, captioned)
  local cap = text.strip_label_spans(text.caption_inlines(t.caption))
  t.caption = pandoc.Caption({})
  return { pandoc.Para(lead_line(word, num, cap)), t }
end

return M
