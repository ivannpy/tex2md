-- Match a float's caption against the .aux \caption lines to get the number the PDF shows.
local S = require("state")
local text = require("strutil")
local M = {}

function M.has_aux(kind)
  for _, f in ipairs(S.floats) do if f.kind == kind then return true end end
  return false
end

-- Number of the next .aux entry of `kind` whose text starts like `caption_text`; nil if none.
-- `label_num` is the number the float's own \\label gives (or nil): it outranks the looser substring match.
function M.aux_number(kind, caption_text, label_num)
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
  local used = S.used[kind]
  -- candidates: the entries from the pointer on, then (the .aux is in placement order) the earlier unused ones
  local order = {}
  for i = ptr, #list do order[#order + 1] = i end
  for i = 1, math.min(ptr - 1, #list) do order[#order + 1] = i end
  local function find(test)
    for _, i in ipairs(order) do
      if not used[i] and test(list[i]) then return i end
    end
  end
  -- longest agreeing prefix first: short probes confuse captions that begin alike ("Unconditional ...")
  local pick
  for _, n in ipairs({ 40, 24, 12 }) do
    local p = key:sub(1, n)
    if p ~= "" and #p >= math.min(n, #key) then
      pick = find(function(f) return f.text:sub(1, #p) == p end)
      if pick then break end
    end
  end
  -- the caption may start with a macro the .aux expanded (\\method~outperforms... -> "UDLM outperforms...")
  if not pick and probe ~= "" and not label_num then
    pick = find(function(f) return f.text:find(probe, 1, true) ~= nil end)
  end
  -- caption text that cannot be compared (all math/macros): trust the order
  if not pick and probe == "" then pick = find(function() return true end) end
  if not pick then return nil end
  used[pick] = true
  if pick >= ptr then S.fptr[kind] = pick + 1 end
  S.flast[kind] = { full = full, num = list[pick].num, idx = pick }
  return list[pick].num
end

return M
