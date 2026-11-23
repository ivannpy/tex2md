-- Display equations: \label{x} becomes \tag{N} with the equation number of the compiled PDF.
local labels = require("labels")
local M = {}

local function tag_math(src)
  -- the author's own \tag{...} wins: a second one is an error for KaTeX (and the .aux number of a tagged
  -- equation is that same text, "$\tau$-leaping", which would put "$" inside the formula)
  -- (\tag{\theequation} is \numberthis, removed later: it does not count)
  local tagged = false
  for arg in src:gmatch("\\tag%*?%s*(%b{})") do
    if not arg:match("^{%s*\\theequation") then tagged = true end
  end
  src = src:gsub("\\label%s*{([^}]*)}", function(id)
    local n = labels.label_number(id)
    if n and not tagged then return "\\tag{" .. n .. "}" end
    return ""
  end)
  return src
end

function M.Math(m)
  if m.mathtype == "DisplayMath" then
    m.text = tag_math(m.text)
  end
  return m
end

return M
