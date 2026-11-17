-- Clean-up pass (runs after the numbering passes): keep the text, drop pandoc's wrappers.
--  * Div / Span become their content (otherwise the Markdown writer emits <div>/<span> HTML);
--  * tables with colspan/rowspan are expanded to plain cells so the writer produces a pipe table
--    (with $math$) instead of an HTML table full of MathML.
local M = {}

local DEFERRED = "tex2md-deferred-table"

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

-- A cell holding only paragraphs becomes one line of inlines (pipe tables need single Plain cells).
local function simple_cell(cell)
  local inlines, ok = pandoc.Inlines({}), true
  for _, b in ipairs(cell.contents) do
    if b.t == "Para" or b.t == "Plain" then
      if #inlines > 0 then inlines:insert(pandoc.Space()) end
      inlines:extend(b.content)
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
      if ok and #cell.contents ~= 1 or (ok and cell.contents[1].t ~= "Plain") then
        row.cells[i] = new
        changed = true
      end
    end
  end
  return changed
end

function M.Table(t)
  local ncols = #t.colspecs
  local changed = false
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
