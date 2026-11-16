-- Lookups in the .aux label table.
local S = require("state")
local M = {}

function M.label_number(id)
  local l = id and S.labels[id]
  return l and l.num or nil
end

-- A \label written before \caption points at the enclosing section: only accept labels of `kind`.
function M.float_number(id, kind)
  local l = id and S.labels[id]
  if l and l.type == kind then return l.num end
  return nil
end

return M
