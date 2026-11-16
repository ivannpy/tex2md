-- Small text and caption helpers shared by the numbering modules.
local M = {}

-- The .aux caption text is raw TeX: drop control-word names on both sides before comparing.
function M.norm_text(s)
  s = s:gsub("\\%a+", "")
  return (s:lower():gsub("[^a-z0-9]", ""))
end

-- 1 -> A, 26 -> Z, 27 -> AA
function M.letter(n)
  local s = ""
  while n > 0 do
    local r = (n - 1) % 26
    s = string.char(65 + r) .. s
    n = (n - 1 - r) // 26
  end
  return s
end

-- A pandoc Caption flattened to inlines.
function M.caption_inlines(caption)
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

-- pandoc leaves \label targets as empty spans; drop them.
function M.strip_label_spans(inlines)
  return inlines:walk({
    Span = function(s)
      if #s.content == 0 then return {} end
    end,
  })
end

return M
