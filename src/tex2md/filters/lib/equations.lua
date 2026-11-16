-- Display equations: \label{x} becomes \tag{N} with the equation number of the compiled PDF.
local labels = require("labels")
local M = {}

local function tag_math(src)
  src = src:gsub("\\label%s*{([^}]*)}", function(id)
    local n = labels.label_number(id)
    if n then return "\\tag{" .. n .. "}" end
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
