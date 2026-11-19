-- Figures and tables: replace each float by a "**Figura N.** caption" / "**Tabla N.** caption" line.
-- The number comes from the .aux caption lines (captions.lua); when the .aux has no data for that kind,
-- from the label table, and as a last resort from a running count.
local S = require("state")
local text = require("strutil")
local labels = require("labels")
local captions = require("captions")
local M = {}

local DEFERRED = "tex2md-deferred-table"

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

-- A table with no text of its own (an image grid laid out with tabular).
local function has_text(block)
  local no_images = block:walk({ Image = function() return {} end })
  return #pandoc.utils.stringify(no_images):gsub("%s", "") > 0
end

local function text_length(block)
  local no_images = block:walk({ Image = function() return {} end })
  return #pandoc.utils.stringify(no_images):gsub("%s", "")
end

-- Text a figure carries besides its images (a prompt in a code box, a quote, a list): worth keeping.
local function substantive(blocks)
  for _, b in ipairs(blocks) do
    if b.t == "CodeBlock" then
      return true
    elseif b.t == "BlockQuote" and substantive(b.content) then
      return true      -- a sample text in a tcolorbox: verbatim inside the quote the box became
    elseif b.t == "Para" or b.t == "Plain" or b.t == "BlockQuote" or b.t == "BulletList"
        or b.t == "OrderedList" or b.t == "DefinitionList" then
      if text_length(b) >= 40 then return true end
    elseif b.t == "Table" then
      -- a grid of images with a few labels is layout; a tabular that boxes whole paragraphs (sample texts) is text
      if text_length(b) >= 150 then return true end
    elseif b.t == "Div" and substantive(b.content) then
      return true
    end
  end
  return false
end

-- The "(a) ..." line the preprocessor writes for the caption of a sub-figure: always kept, however short.
local function is_subcaption(b)
  if (b.t ~= "Para" and b.t ~= "Plain") or #b.content == 0 or b.content[1].t ~= "Emph" then return false end
  return pandoc.utils.stringify(b.content[1]):match("^%(%w+%)") ~= nil
end

local function subcaptions(blocks, out)
  for _, b in ipairs(blocks) do
    if is_subcaption(b) then
      out:insert(b)
    elseif b.t == "Div" then
      subcaptions(b.content, out)
    end
  end
  return out
end

-- A figure that holds tables: the tables with text, and around them the text of the figure (sub-captions,
-- notes). `first` receives the first table.
local function tables_and_text(blocks, out, state)
  for _, b in ipairs(blocks) do
    if b.t == "Table" then
      if has_text(b) then
        state.first = state.first or b
        out:insert(b)
      end
    elseif b.t == "Div" then
      tables_and_text(b.content, out, state)
    elseif text_length(b) > 0 then
      out:insert(b)
    end
  end
  return out
end

function M.Figure(fig)
  S.stats.figures = S.stats.figures + 1
  local num = labels.float_number(fig.identifier, "figure")
  local has_table = false
  fig.content:walk({ Table = function(t) if has_text(t) then has_table = true end end })
  local lab = fig.identifier ~= "" and S.labels[fig.identifier] or nil
  -- an image grid laid out with tabular is still a figure when its label says so
  if has_table and not (lab and lab.type == "figure") then
    local state = {}
    local out = tables_and_text(fig.content, pandoc.Blocks({}), state)
    local first = state.first
    if first then
      -- A table environment wrapped as a figure: hand its caption to the first table and let the
      -- Table pass print it (as "Figura N" when the .aux lists it among the figures).
      S.stats.figures = S.stats.figures - 1
      if #(first.caption.long or {}) == 0 then first.caption = fig.caption end
      if first.identifier == "" then first.identifier = fig.identifier end
      local fnum = has_caption(first.caption)
        and captions.aux_number("figure", pandoc.utils.stringify(first.caption.long)) or nil
      if fnum then
        first.attr = pandoc.Attr(first.attr.identifier, first.attr.classes, { tex2md_fignum = fnum })
      end
      return out
    end
  end

  local captioned = has_caption(fig.caption)
  -- A table without tabular was turned into a figure by the preprocessor and its caption marked.
  -- Tables are numbered in the Table pass (document order among the tables), so hand it over.
  if captioned and pandoc.utils.stringify(fig.caption.long):sub(1, 11) == "TEX2MDTABLE" then
    local cap = text.strip_label_spans(text.caption_inlines(fig.caption))
    return pandoc.Div({ pandoc.Para(cap) }, pandoc.Attr("", { DEFERRED }))
  end
  local word, counter, kind = "Figura", "fig_count", "figure"
  if captioned and captions.caption_kind(pandoc.utils.stringify(fig.caption.long)) == "table" then
    word, counter, kind = "Tabla", "tab_count", "table"
  end
  if captioned then
    num = captions.aux_number(kind, pandoc.utils.stringify(fig.caption.long), kind == "figure" and num or nil)
      or (kind == "figure" and num or nil)
  else
    num = nil  -- a panel without its own caption carries its parent's label: no number
  end
  num = resolve_number(counter, kind, num, captioned)
  local cap = text.strip_label_spans(text.caption_inlines(fig.caption))
  local line = pandoc.Para(lead_line(word, num, cap))
  local out = pandoc.Blocks({})
  if substantive(fig.content) then
    -- the tables kept here are part of the figure: the Table pass must not print a "Tabla" line for them
    out:extend(fig.content:walk({ Table = function(t)
      t.attr = pandoc.Attr(t.attr.identifier, t.attr.classes, { tex2md_in_figure = "1" })
      return t
    end }))
  else
    subcaptions(fig.content, out)
  end
  if #out == 0 then return line end
  out:insert(line)
  return out
end

-- Table pass: a marked caption that the Figure pass deferred (see above).
function M.DeferredTable(d)
  if not d.classes:includes(DEFERRED) then return nil end
  local cap = d.content[1].content
  local num = captions.aux_number("table", (pandoc.utils.stringify(cap):gsub("^TEX2MDTABLE", "")))
  cap:remove(1)  -- the marker
  if cap[1] and cap[1].t == "Space" then cap:remove(1) end
  -- a table turned into a figure still counts with the figures when there is no .aux data
  num = resolve_number("fig_count", "table", num, true)
  return pandoc.Para(lead_line("Tabla", num, cap))
end

-- A tabular inside a cell of another tabular (two lines stacked in one cell) is part of that cell, not a
-- table of its own: mark it, so that it gets neither a caption line nor a number.
function M.MarkNested(t)
  return t:walk({ Table = function(inner)
    inner.attr = pandoc.Attr(inner.attr.identifier, inner.attr.classes, { tex2md_nested = "1" })
    return inner
  end })
end

function M.Table(t)
  -- printed with its figure (see M.Figure), or part of a cell of another table
  if t.attr.attributes["tex2md_in_figure"] or t.attr.attributes["tex2md_nested"] then return nil end
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
      num = captions.aux_number("table", pandoc.utils.stringify(t.caption.long), num) or num
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
