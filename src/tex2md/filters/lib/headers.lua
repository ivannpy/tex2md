-- Section numbers: taken from the .aux table of contents (matched by title, in order) or from the heading's label,
-- counted otherwise. Counting alone goes wrong when pandoc shifts the levels (a \part) or a heading is lost.
local S = require("state")
local text = require("strutil")
local labels = require("labels")
local M = {}

local function header_number(level)
  if level > 3 then return nil end
  local c = S.counters
  c[level] = c[level] + 1
  for i = level + 1, 3 do c[i] = 0 end
  local parts = {}
  for i = 1, level do
    if c[i] == 0 then return nil end
    if i == 1 and S.in_appendix then
      parts[#parts + 1] = text.letter(c[1])
    else
      parts[#parts + 1] = tostring(c[i])
    end
  end
  return table.concat(parts, ".")
end

-- Keep the running counters aligned with a number taken from the .aux.
local function resync(num)
  local c = S.counters
  local i = 0
  for part in string.gmatch(num, "[^%.]+") do
    i = i + 1
    if i > 3 then break end
    if part:match("^%d+$") then
      c[i] = tonumber(part)
    elseif part:match("^%u+$") then
      local v = 0
      for ch in part:gmatch(".") do v = v * 26 + (ch:byte() - 64) end
      c[i] = v
    end
  end
  for j = i + 1, 3 do c[j] = 0 end
end

local function unnumbered(classes)
  for _, c in ipairs(classes) do
    if c == "unnumbered" then return true end
  end
  return false
end

local TOC_LOOKAHEAD = 12      -- toc entries a header may skip (headings the conversion lost or merged)

-- The number of the next table-of-contents entry whose title starts like this header's; nil if none.
local function toc_number(h)
  local key = text.norm_text(pandoc.utils.stringify(h.content)):sub(1, 40)
  if key == "" then return nil end
  for i = S.sptr, math.min(S.sptr + TOC_LOOKAHEAD, #S.sections) do
    local e = S.sections[i]
    local n = math.min(#key, #e.text)
    -- the level must agree too (once known): a \paragraph often repeats the title of a later subsection
    local level_ok = S.level_shift == nil or h.level - e.level == S.level_shift
    if level_ok and e.text ~= "" and (e.text == key or (n >= 12 and e.text:sub(1, n) == key:sub(1, n))) then
      S.level_shift = S.level_shift or h.level - e.level
      S.sptr = i + 1
      return e.num
    end
  end
end

function M.Header(h)
  if unnumbered(h.classes) then return h end
  -- above the level of \section (known once a header matched the toc): a \part, which the toc does not number
  if S.level_shift and h.level - S.level_shift < 1 then return h end
  local num = toc_number(h) or labels.label_number(h.identifier)
  local auto = header_number(h.level)
  if num then resync(num) else num = auto end
  if num then
    h.content:insert(1, pandoc.Space())
    h.content:insert(1, pandoc.Str(num))
  end
  return h
end

-- The preprocessor leaves a marker paragraph where \appendix started: sections become A, B, ...
function M.Para(p)
  if pandoc.utils.stringify(p) == "TEX2MDAPPENDIXMARKER" then
    S.in_appendix = true
    S.counters = { 0, 0, 0 }
    return {}
  end
end

return M
