-- Clean-up pass (runs after the numbering passes): keep the text, drop pandoc's wrappers.
--  * Div / Span become their content (otherwise the Markdown writer emits <div>/<span> HTML);
--  * tables with colspan/rowspan are expanded to plain cells so the writer produces a pipe table
--    (with $math$) instead of an HTML table full of MathML.
local labels = require("labels")
local M = {}

local DEFERRED = "tex2md-deferred-table"

-- Languages a Markdown viewer highlights; the name of a listings style ("PythonFuncColor") is not one.
local LANGUAGES = { python = true, bash = true, sh = true, shell = true, c = true, cpp = true, java = true,
  javascript = true, json = true, yaml = true, latex = true, tex = true, sql = true, r = true, julia = true,
  matlab = true, lua = true, go = true, rust = true, html = true, xml = true, text = true, console = true }

-- A listing: pandoc writes its options as attributes of the fence (``` {.python language="Python" caption="..."}).
-- Keep the language only, and print the caption as a line of its own, numbered like the PDF when it has a label.
function M.CodeBlock(cb)
  local caption, id = cb.attributes["caption"], cb.identifier
  local lang = (cb.classes[1] or cb.attributes["language"] or ""):lower()
  if lang:find("python", 1, true) then lang = "python" end
  local plain = id == "" and #cb.attributes == 0 and #cb.classes <= 1 and (lang == "" or LANGUAGES[lang])
  if plain and (cb.classes[1] or "") == lang then return nil end
  cb.attr = pandoc.Attr("", LANGUAGES[lang] and { lang } or {})
  if not caption or caption == "" then return cb end
  local num = labels.label_number(id)
  local lead = pandoc.Inlines({ pandoc.Strong(pandoc.Str("Listado" .. (num and (" " .. num) or "") .. ".")), pandoc.Space() })
  lead:extend(pandoc.utils.blocks_to_inlines(pandoc.read(caption, "latex").blocks))
  return { pandoc.Para(lead), cb }
end

function M.Div(d)
  if d.classes:includes(DEFERRED) then return nil end
  return d.content
end

function M.Span(s)
  return s.content
end

-- \\textsc{...}: the Markdown writer would emit <span class="smallcaps">
function M.SmallCaps(s)
  return s.content
end

local function empty_cell()
  return pandoc.Cell({}, pandoc.AlignDefault, 1, 1)
end

-- Rewrite the rows of one section so that no cell spans several columns or rows.
local function flatten_rows(rows, ncols)
  local carry = {}                       -- column -> rows still covered by a cell from above
  local out = {}
  for _, row in ipairs(rows) do
    local cells, col = {}, 1
    local function fill_carried()
      while col <= ncols and (carry[col] or 0) > 0 do
        carry[col] = carry[col] - 1
        cells[#cells + 1] = empty_cell()
        col = col + 1
      end
    end
    for _, cell in ipairs(row.cells) do
      fill_carried()
      local cs, rs = cell.col_span or 1, cell.row_span or 1
      cells[#cells + 1] = pandoc.Cell(cell.contents, cell.alignment, 1, 1)
      for k = 0, cs - 1 do
        if rs > 1 then carry[col + k] = rs - 1 end
        if k > 0 then cells[#cells + 1] = empty_cell() end
      end
      col = col + cs
    end
    fill_carried()
    out[#out + 1] = pandoc.Row(cells, row.attr)
  end
  return out
end

local function has_span(rows)
  for _, row in ipairs(rows) do
    for _, cell in ipairs(row.cells) do
      if (cell.col_span or 1) > 1 or (cell.row_span or 1) > 1 then return true end
    end
  end
  return false
end

-- A line break inside a cell: pipe tables cannot hold one, but they take an inline <br>
local function html_breaks(inlines)
  return inlines:walk({ LineBreak = function() return pandoc.RawInline("html", "<br>") end })
end

-- The inlines of a block that can sit on one line: a paragraph, a code block (one <code> per line), or a table
-- nested in a cell (a one-column tabular used to stack the lines of a label) read row by row. nil for anything else.
local function line_of(block)
  if block.t == "Para" or block.t == "Plain" then return html_breaks(block.content) end
  if block.t == "CodeBlock" then
    local inlines = pandoc.Inlines({})
    for line in (block.text .. "\n"):gmatch("(.-)\n") do
      if #inlines > 0 then inlines:insert(pandoc.RawInline("html", "<br>")) end
      if line ~= "" then inlines:insert(pandoc.Code(line)) end
    end
    return inlines
  end
  if block.t ~= "Table" then return nil end
  local inlines = pandoc.Inlines({})
  local function add_rows(rows)
    for _, row in ipairs(rows) do
      for _, cell in ipairs(row.cells) do
        for _, b in ipairs(cell.contents) do
          local part = line_of(b)
          if part == nil then return false end
          if #part > 0 then
            if #inlines > 0 then inlines:insert(pandoc.Space()) end
            inlines:extend(part)
          end
        end
      end
    end
    return true
  end
  if not add_rows(block.head.rows) then return nil end
  for _, body in ipairs(block.bodies) do
    if not (add_rows(body.head) and add_rows(body.body)) then return nil end
  end
  if not add_rows(block.foot.rows) then return nil end
  return inlines
end

-- A cell holding only paragraphs becomes one line of inlines (pipe tables need single Plain cells).
local function simple_cell(cell)
  local inlines, ok = pandoc.Inlines({}), true
  for _, b in ipairs(cell.contents) do
    local part = line_of(b)
    if part then
      if #inlines > 0 and #part > 0 then inlines:insert(pandoc.Space()) end
      inlines:extend(part)
    else
      ok = false
    end
  end
  if not ok then return cell, false end
  return pandoc.Cell({ pandoc.Plain(inlines) }, cell.alignment, cell.row_span, cell.col_span), true
end

local function simplify_rows(rows)
  local changed = false
  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      local new, ok = simple_cell(cell)
      if ok and (#cell.contents ~= 1 or cell.contents[1].t ~= "Plain" or new.contents[1] ~= cell.contents[1]) then
        row.cells[i] = new
        changed = true
      end
    end
  end
  return changed
end

-- The only cell of a one-column, one-row table, or nil. Such a "table" is a box around some text.
local function single_cell(t)
  if #t.colspecs ~= 1 then return nil end
  local rows = {}
  for _, r in ipairs(t.head.rows) do rows[#rows + 1] = r end
  for _, b in ipairs(t.bodies) do
    for _, r in ipairs(b.head) do rows[#rows + 1] = r end
    for _, r in ipairs(b.body) do rows[#rows + 1] = r end
  end
  for _, r in ipairs(t.foot.rows) do rows[#rows + 1] = r end
  if #rows ~= 1 or #rows[1].cells ~= 1 then return nil end
  return rows[1].cells[1]
end

local function row_is_empty(row)
  for _, cell in ipairs(row.cells) do
    local text = pandoc.utils.stringify(pandoc.Div(cell.contents))
    if text:find("%S") then return false end
  end
  return true
end

-- Empty (or missing) header and a first body row with something in it: that row becomes the header.
local function promote_first_row(t)
  local body = t.bodies[1]
  if not body or #body.body == 0 or #body.head > 0 then return false end
  if #t.head.rows > 1 or (#t.head.rows == 1 and not row_is_empty(t.head.rows[1])) then return false end
  local first = 1                      -- blank rows at the top are spacing the authors typed (& & \\)
  while first <= #body.body and row_is_empty(body.body[first]) do first = first + 1 end
  if first > #body.body then return false end
  local rest = {}
  for i = first + 1, #body.body do rest[#rest + 1] = body.body[i] end
  t.head.rows = { body.body[first] }
  body.body = rest
  return true
end

function M.Table(t)
  local ncols = #t.colspecs
  local changed = false
  -- a box around some text (fbox + minipage inside a one-column tabular): the text itself, not a one-cell table
  local cell = single_cell(t)
  if cell and #cell.contents > 0 then
    local blocks = pandoc.Blocks({})
    for _, b in ipairs(cell.contents) do
      blocks:insert(b.t == "Plain" and pandoc.Para(b.content) or b)
    end
    return blocks
  end
  -- a table with an id/class (from its \\label) can only be written as HTML
  if t.attr.identifier ~= "" or #t.attr.classes > 0 or #t.attr.attributes > 0 then
    t.attr = pandoc.Attr()
    changed = true
  end
  if has_span(t.head.rows) then
    t.head.rows = flatten_rows(t.head.rows, ncols); changed = true
  end
  for _, body in ipairs(t.bodies) do
    if has_span(body.body) or has_span(body.head) then
      body.body = flatten_rows(body.body, ncols)
      body.head = flatten_rows(body.head, ncols)
      changed = true
    end
    if simplify_rows(body.body) then changed = true end
  end
  if has_span(t.foot.rows) then
    t.foot.rows = flatten_rows(t.foot.rows, ncols); changed = true
  end
  if simplify_rows(t.head.rows) then changed = true end
  -- a pipe table needs a header row and pandoc writes an empty one when the LaTeX table has none: the first
  -- row of the body is the best header there is
  if promote_first_row(t) then changed = true end
  -- a pipe table has a single header row: the others become the first body rows
  if #t.head.rows > 1 and #t.bodies > 0 then
    local extra = {}
    for i = 2, #t.head.rows do extra[#extra + 1] = t.head.rows[i] end
    t.head.rows = { t.head.rows[1] }
    for _, r in ipairs(t.bodies[1].body) do extra[#extra + 1] = r end
    t.bodies[1].body = extra
    changed = true
  end
  if changed then return t end
end

return M
