-- Match a float's caption against the .aux \caption lines to get the number the PDF shows.
local S = require("state")
local text = require("strutil")
local M = {}

function M.has_aux(kind)
  for _, f in ipairs(S.floats) do if f.kind == kind then return true end end
  return false
end

-- Number of the next .aux entry of `kind` whose text starts like `caption_text`; nil if none.
function M.aux_number(kind, caption_text)
  local full = text.norm_text(caption_text)
  local key = full:sub(1, 40)
  local list, probe = {}, key:sub(1, 12)
  for _, f in ipairs(S.floats) do if f.kind == kind then list[#list + 1] = f end end
  -- a float with several tabulars repeats its whole caption: same number
  local last = S.flast[kind]
  if last and last.full == full and full ~= "" then
    -- unless the .aux really has another float with the same caption right after it
    local nxt = list[last.idx + 1]
    if not (nxt and probe ~= "" and nxt.text:sub(1, #probe) == probe) then return last.num end
  end
  local ptr = S.fptr[kind]
  local pick
  for i = ptr, #list do
    if probe ~= "" and list[i].text:sub(1, #probe) == probe then pick = i; break end
  end
  -- caption text that cannot be compared (all math/macros): trust the order
  if not pick and probe == "" and ptr <= #list then pick = ptr end
  if not pick then return nil end
  S.fptr[kind] = pick + 1
  S.flast[kind] = { full = full, num = list[pick].num, idx = pick }
  return list[pick].num
end

return M
