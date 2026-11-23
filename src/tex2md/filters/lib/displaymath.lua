-- Display equations on their own paragraph.
--
-- In LaTeX a display equation sits inside the paragraph (theorem text, list items, emphasis). Pandoc keeps it
-- there as an inline DisplayMath and writes "text $$...$$ text" on one line, or "*Theorem text $$...$$*".
-- Many Markdown viewers (MarginNote, Obsidian, KaTeX plug-ins) only render $$ blocks that start a line, and
-- an emphasis that spans the block breaks. Here every paragraph is split at its display equations: text
-- before / equation / text after, each its own block, with emphasis closed before the equation and reopened
-- after. Inside a block quote or a list item the writer then prefixes and indents each block correctly.
-- Table cells are left alone: a pipe-table cell must stay on one line.
local M = {}

-- inline containers that can be cut in two around an equation
local WRAP = { Emph = true, Strong = true, Underline = true, Strikeout = true, SmallCaps = true, Span = true }

local function has_display(inlines)
  for _, el in ipairs(inlines) do
    if el.t == "Math" and el.mathtype == "DisplayMath" then return true end
    if WRAP[el.t] and has_display(el.content) then return true end
  end
  return false
end

local function rewrap(el, content)
  local c = el:clone()
  c.content = content
  return c
end

-- Split inlines into a list of segments: { inlines } for text, { math = Math } for an equation.
local function split(inlines)
  local segs = { { inl = pandoc.Inlines({}) } }
  for _, el in ipairs(inlines) do
    if el.t == "Math" and el.mathtype == "DisplayMath" then
      segs[#segs + 1] = { math = el }
      segs[#segs + 1] = { inl = pandoc.Inlines({}) }
    elseif WRAP[el.t] and has_display(el.content) then
      for _, s in ipairs(split(el.content)) do
        if s.math then
          segs[#segs + 1] = s
          segs[#segs + 1] = { inl = pandoc.Inlines({}) }
        elseif #s.inl > 0 then
          -- the first piece continues the current text; the others start after an equation
          segs[#segs].inl:insert(rewrap(el, s.inl))
        end
      end
    else
      segs[#segs].inl:insert(el)
    end
  end
  return segs
end

local function blank(el) return el.t == "Space" or el.t == "SoftBreak" or el.t == "LineBreak" end

local function trim(inl)
  while #inl > 0 and blank(inl[1]) do inl:remove(1) end
  while #inl > 0 and blank(inl[#inl]) do inl:remove(#inl) end
  -- an emphasis left with only blanks inside (the space before the equation)
  for i = #inl, 1, -1 do
    local el = inl[i]
    if WRAP[el.t] then
      trim(el.content)
      if #el.content == 0 then inl:remove(i) end
    end
  end
  while #inl > 0 and blank(inl[1]) do inl:remove(1) end
  while #inl > 0 and blank(inl[#inl]) do inl:remove(#inl) end
  return inl
end

-- the text right after an equation that is only its punctuation (". " or ",") goes into the equation
local function leading_punct(inl)
  local first = inl[1]
  while first and WRAP[first.t] do first = first.content[1] end
  if first and first.t == "Str" then
    local p, rest = first.text:match("^([%.,;:]+)(.*)$")
    if p then
      first.text = rest
      return p
    end
  end
  return nil
end

local function add_punct(math, p)
  local text = math.text:gsub("%s+$", "")
  local body, env = text:match("^(.-)(\\end%s*{[%a%*]+}%s*)$")
  if body then
    math.text = body:gsub("%s+$", "") .. p .. "\n" .. env
  else
    math.text = text .. p
  end
end

local function drop_empty_str(inl)
  for i = #inl, 1, -1 do
    local el = inl[i]
    if el.t == "Str" and el.text == "" then
      inl:remove(i)
    elseif WRAP[el.t] then
      drop_empty_str(el.content)
      if #el.content == 0 then inl:remove(i) end
    end
  end
  return inl
end

local function split_block(b)
  local segs = split(b.content)
  local out = pandoc.Blocks({})
  -- first the punctuation that follows each equation goes into it (a block built from an element copies it)
  for i, s in ipairs(segs) do
    if not s.math then
      s.inl = trim(s.inl)
      if i > 1 and segs[i - 1].math and #s.inl > 0 then
        local p = leading_punct(s.inl)
        if p then
          add_punct(segs[i - 1].math, p)
          s.inl = trim(drop_empty_str(s.inl))
        end
      end
    end
  end
  for _, s in ipairs(segs) do
    if s.math then
      out:insert(pandoc.Para({ s.math }))
    elseif #s.inl > 0 then
      out:insert(b.t == "Plain" and pandoc.Plain(s.inl) or pandoc.Para(s.inl))
    end
  end
  return out
end

local function walk(blocks)
  local out = pandoc.Blocks({})
  for _, b in ipairs(blocks) do
    if (b.t == "Para" or b.t == "Plain") and has_display(b.content) then
      out:extend(split_block(b))
    elseif b.t == "BlockQuote" or b.t == "Div" or b.t == "Figure" then
      b.content = walk(b.content)
      out:insert(b)
    elseif b.t == "BulletList" or b.t == "OrderedList" then
      for i, item in ipairs(b.content) do b.content[i] = walk(item) end
      out:insert(b)
    elseif b.t == "DefinitionList" then
      for _, item in ipairs(b.content) do
        for j, def in ipairs(item[2]) do item[2][j] = walk(def) end
      end
      out:insert(b)
    else
      out:insert(b)   -- Table: cells stay on one line; code and raw blocks have no math
    end
  end
  return out
end

-- Inline math written over several lines: in a block quote the writer would put "> " in the middle of the
-- formula. One line, and no blank at either end ("$ x$" is not math for most readers).
function M.Math(m)
  if m.mathtype == "InlineMath" then
    local t = m.text
    if t:find("\n") then
      -- a TeX comment ("%" to the end of the line, often a bare "%" after a macro body) would swallow the
      -- rest of the formula once it is on one line
      local lines = {}
      for line in (t .. "\n"):gmatch("(.-)\n") do
        local i, cut = 1, nil
        while i <= #line do
          local c = line:sub(i, i)
          if c == "\\" then i = i + 2
          elseif c == "%" then cut = i; break
          else i = i + 1 end
        end
        lines[#lines + 1] = cut and line:sub(1, cut - 1) or line
      end
      t = table.concat(lines, "\n")
    end
    t = t:gsub("%s*\n%s*", " "):gsub("^%s+", "")
    -- trailing blanks, and a trailing control space "\ " (but not "\\ ", a line break)
    t = t:gsub("%s+$", "")
    while t:match("[^\\]\\$") or t == "\\" do
      t = t:sub(1, -2):gsub("%s+$", "")
    end
    m.text = t
    return m
  end
end

function M.Pandoc(doc)
  doc.blocks = walk(doc.blocks)
  return doc
end

return M
