#!/usr/bin/env ruby

total_lines = 0
total_score = 0

ARGF.each do |line|
  langid, text = line.split("\t")
  /\('(?<lang>..)', (?<score>.+)\)/ =~ langid
  score = score.to_f
  text.strip!
  if text.length != 0
    total_lines += 1
    total_score += score
  end
end

puts (total_score / total_lines)