-- tex2md pandoc filter.
-- Reads the label table produced from the compiled .aux (metadata field
-- `tex2md_labels`, a JSON file path) and uses it to number headers, figures,
-- tables and equations the same way the compiled PDF does.

local labels = {}
local in_appendix = false
local counters = {0, 0, 0}
local stats = {figures = 0, tables = 0}

local function read_labels(meta)
  local path = meta.tex2md_labels and pandoc.utils.stringify(meta.tex2md_labels)
  if not path or path == "" then return end
  local fh = io.open(path, "r")
  if not fh then return end
  labels = pandoc.json.decode(fh:read("a")) or {}
  fh:close()
end

-- Real figure/table numbers from the \\caption lines of the .aux, in document order.
local floats = {}
local fptr = { figure = 1, table = 1 }
local flast = {}

local function read_floats(meta)
  local path = meta.tex2md_floats and pandoc.utils.stringify(meta.tex2md_floats)
  if not path or path == "" then return end
  local fh = io.open(path, "r")
  if not fh then return end
  floats = pandoc.json.decode(fh:read("a")) or {}
  fh:close()
end

local function norm_text(s)
  -- the .aux text is raw TeX: drop control-word names on both sides before comparing
  s = s:gsub("\\%a+", "")
  return (s:lower():gsub("[^a-z0-9]", ""))
end

local function has_aux(kind)
  for _, f in ipairs(floats) do if f.kind == kind then return true end end
  return false
end

local function aux_number(kind, caption_text)
  local full = norm_text(caption_text)
  local key = full:sub(1, 40)
  -- a float with several tabulars repeats its whole caption: same number
  local list, probe = {}, key:sub(1, 12)
  for _, f in ipairs(floats) do if f.kind == kind then list[#list + 1] = f end end
  if flast[kind] and flast[kind].full == full and full ~= "" then
    -- unless the .aux really has another float with the same caption right after it
    local nxt = list[flast[kind].idx + 1]
    if not (nxt and probe ~= "" and nxt.text:sub(1, #probe) == probe) then return flast[kind].num end
  end
  local ptr = fptr[kind]
  local pick
  for i = ptr, #list do
    if probe ~= "" and list[i].text:sub(1, #probe) == probe then pick = i; break end
  end
  -- caption text that cannot be compared (all math/macros): trust the order
  if not pick and probe == "" and ptr <= #list then pick = ptr end
  if not pick then return nil end
  fptr[kind] = pick + 1
  flast[kind] = { full = full, num = list[pick].num, idx = pick }
  return list[pick].num
end

local function label_number(id)
  local l = id and labels[id]
  return l and l.num or nil
end

-- a \label written before \caption points at the enclosing section: ignore those
local function float_number(id, kind)
  local l = id and labels[id]
  if l and l.type == kind then return l.num end
  return nil
end

local function letter(n)
  -- 1 -> A, 26 -> Z, 27 -> AA
  local s = ""
  while n > 0 do
    local r = (n - 1) % 26
    s = string.char(65 + r) .. s
    n = (n - 1 - r) // 26
  end
  return s
end

local function header_number(level)
  if level > 3 then return nil end
  counters[level] = counters[level] + 1
  for i = level + 1, 3 do counters[i] = 0 end
  local parts = {}
  for i = 1, level do
    if counters[i] == 0 then return nil end
    if i == 1 and in_appendix then
      parts[#parts + 1] = letter(counters[1])
    else
      parts[#parts + 1] = tostring(counters[i])
    end
  end
  return table.concat(parts, ".")
end

local function resync(level, num)
  -- Keep the running counters aligned with a number taken from the .aux.
  local i = 0
  for part in string.gmatch(num, "[^%.]+") do
    i = i + 1
    if i > 3 then break end
    if part:match("^%d+$") then
      counters[i] = tonumber(part)
    elseif part:match("^%u+$") then
      local v = 0
      for c in part:gmatch(".") do v = v * 26 + (c:byte() - 64) end
      counters[i] = v
    end
  end
  for j = i + 1, 3 do counters[j] = 0 end
end

local function caption_inlines(caption)
  local blocks = caption and caption.long or {}
  local out = pandoc.Inlines({})
  for _, b in ipairs(blocks) do
    if b.content then
      if #out > 0 then out:insert(pandoc.Space()) end
      out:extend(b.content)
    end
  end
  return out
end

local function strip_label_spans(inlines)
  -- pandoc leaves \label targets as empty spans; drop them.
  return inlines:walk({
    Span = function(s)
      if #s.content == 0 then return {} end
    end,
  })
end

local function tag_math(text)
  text = text:gsub("\\label%s*{([^}]*)}", function(id)
    local n = label_number(id)
    if n then return "\\tag{" .. n .. "}" end
    return ""
  end)
  return text
end

local function unnumbered(classes)
  for _, c in ipairs(classes) do
    if c == "unnumbered" then return true end
  end
  return false
end

local fig_count, tab_count = 0, 0
local filter = {}

filter.Meta = function(meta)
  read_labels(meta)
  read_floats(meta)
  return meta
end

filter.Header = function(h)
  if unnumbered(h.classes) then return h end
  local num = label_number(h.identifier)
  local auto = header_number(h.level)
  if num then resync(h.level, num) else num = auto end
  if num then
    h.content:insert(1, pandoc.Space())
    h.content:insert(1, pandoc.Str(num))
  end
  return h
end

filter.Para = function(p)
  local text = pandoc.utils.stringify(p)
  if text == "TEX2MDAPPENDIXMARKER" then
    in_appendix = true
    counters = {0, 0, 0}
    return {}
  end
end

filter.Figure = function(fig)
  stats.figures = stats.figures + 1
  local num = float_number(fig.identifier, "figure")
  local tables = {}
  fig.content:walk({ Table = function(t) tables[#tables + 1] = t end })
  local lab = fig.identifier ~= "" and labels[fig.identifier] or nil
  -- an image grid laid out with tabular is still a figure when its label says so
  if #tables > 0 and not (lab and lab.type == "figure") then
    -- A table environment wrapped as a figure: hand its caption to the
    -- first table and let the Table pass number it.
    stats.figures = stats.figures - 1
    local first = tables[1]
    if #(first.caption.long or {}) == 0 then first.caption = fig.caption end
    if first.identifier == "" then first.identifier = fig.identifier end
    local fnum = #(first.caption.long or {}) > 0
      and aux_number("figure", pandoc.utils.stringify(first.caption.long)) or nil
    if fnum then
      first.attr = pandoc.Attr(first.attr.identifier, first.attr.classes, { tex2md_fignum = fnum })
    end
    return tables
  end
  local captioned = #(fig.caption.long or {}) > 0
  local word, kind = "Figura", "figure"
  local marked = captioned and pandoc.utils.stringify(fig.caption.long):sub(1, 11) == "TEX2MDTABLE"
  if marked then word, kind = "Tabla", "table"; num = nil end
  if captioned then
    num = aux_number(kind, (pandoc.utils.stringify(fig.caption.long):gsub("^TEX2MDTABLE", ""))) or num
  else
    num = nil  -- a panel without its own caption carries its parent's label: no number
  end
  if num then
    local n = tonumber(num)
    if n then fig_count = n end
  elseif captioned and not has_aux(kind) then
    fig_count = fig_count + 1
    num = tostring(fig_count)
  end
  local cap = strip_label_spans(caption_inlines(fig.caption))
  if marked then
    cap:remove(1)
    if cap[1] and cap[1].t == "Space" then cap:remove(1) end
  end
  local lead = pandoc.Inlines({ pandoc.Strong(pandoc.Str(word .. (num and (" " .. num) or "") .. ".")) })
  if #cap > 0 then lead:insert(pandoc.Space()); lead:extend(cap) end
  return pandoc.Para(lead)
end

filter.Table = function(t)
  stats.tables = stats.tables + 1
  local num = float_number(t.identifier, "table")
  if not num then
    -- pandoc puts the \label of a table on a span inside the caption
    pandoc.walk_block(pandoc.Div(t.caption.long or {}), {
      Span = function(s) if s.identifier ~= "" then num = num or float_number(s.identifier, "table") end end,
    })
  end
  local captioned = #(t.caption.long or {}) > 0
  local word = "Tabla"
  if captioned then
    local text = pandoc.utils.stringify(t.caption.long)
    local fnum = t.attr.attributes["tex2md_fignum"]
    if fnum then
      num, word = fnum, "Figura"
    else
      num = aux_number("table", pandoc.utils.stringify(t.caption.long)) or num
    end
  else
    num = nil
  end
  if num then
    local n = tonumber(num)
    if n then tab_count = n end
  elseif captioned and not has_aux("table") then
    tab_count = tab_count + 1
    num = tostring(tab_count)
  end
  local cap = strip_label_spans(caption_inlines(t.caption))
  t.caption = pandoc.Caption({})
  local lead = pandoc.Inlines({ pandoc.Strong(pandoc.Str(word .. (num and (" " .. num) or "") .. ".")) })
  if #cap > 0 then lead:insert(pandoc.Space()); lead:extend(cap) end
  return { pandoc.Para(lead), t }
end

filter.Math = function(m)
  if m.mathtype == "DisplayMath" then
    m.text = tag_math(m.text)
  end
  return m
end

filter.Image = function(_) return {} end

filter.Pandoc = function(doc)
  local blocks = pandoc.Blocks({})
  if doc.meta.abstract then
    blocks:insert(pandoc.Header(2, "Abstract", pandoc.Attr("", { "unnumbered" })))
    local abs = doc.meta.abstract
    if pandoc.utils.type(abs) == "Blocks" then
      blocks:extend(abs)
    else
      blocks:insert(pandoc.Para(abs))
    end
  end
  blocks:extend(doc.blocks)
  doc.blocks = blocks
  io.stderr:write(string.format("tex2md-stats figures=%d tables=%d\n", stats.figures, stats.tables))
  return doc
end

return {
  { Meta = filter.Meta },
  { Para = filter.Para, Header = filter.Header },
  { Figure = filter.Figure },
  { Table = filter.Table, Math = filter.Math, Image = filter.Image },
  { Pandoc = filter.Pandoc },
}
