# 用 Windows 自带中文语音为视频生成七段旁白（仅生成音频文件，不修改其他内容）
$ErrorActionPreference = 'Stop'
$dir = Join-Path $PSScriptRoot 'voice'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$log = Join-Path $dir 'tts_log.txt'
$texts = @(
  '数据中台里，公司、矿山和指标都已经能查。现在业务还想看清它们之间的关联，比如哪家公司运营哪些矿山。前提是，原有数据和业务流程都不改。',
  '做法是在中台之外加一套知识图谱，把对象和关系连成一张网。外部同步程序从中台只读获取，只取不改；在外部完成转换和ID对应，再写入 Neo 4 J，一种专门存关系的图数据库，供查询服务和页面使用。',
  '同一座矿山，图谱里是 M I N 零零零一，中台里是 M 零零一。两个ID不必相同，用对应表连起来。程序先按可靠编号和已确认的对应自动匹配；拿不准的标为待核，人工确认后保存，以后直接复用。',
  '关系怎么来？中台已有的关系直接保留；字段含义明确的，比如“运营单位”，就按规则生成“运营”关系，并记下来源和审核状态。但要注意：名称相似，不能证明是同一对象；“关联”也不能直接当成“拥有”。',
  '查示例甲公司关联矿山二零二四年的锑产量：图谱找到矿山，给出中台对象ID和指标ID；再调用中台原指标接口，也就是原有取数通道，按原统计规则和权限返回数值。缺数显示“暂无数据”，无权限显示“无权限”。',
  '新增数据分两种。已有矿山新增年度数值，沿用原对象ID；中台出现新矿山，确认后由外部程序生成新图谱ID并保存对应，有明确依据再建立关系。',
  '先用四份 C S V 表格文件搭练习图谱，验收后换成真实数据，再接入只读接口和联合查询服务。'
)
"start $(Get-Date)" | Out-File $log -Encoding utf8
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo })
"SAPI voices:" | Out-File $log -Append -Encoding utf8
$voices | ForEach-Object { "  $($_.Name) | $($_.Culture) | $($_.Gender)" } | Out-File $log -Append -Encoding utf8
$zh = $voices | Where-Object { $_.Culture.Name -like 'zh-*' } | Select-Object -First 1
if ($zh) {
  $s.SelectVoice($zh.Name); $s.Rate = 1; $s.Volume = 100
  $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  for ($i = 0; $i -lt $texts.Count; $i++) {
    $f = Join-Path $dir ("s{0}.wav" -f ($i + 1))
    $s.SetOutputToWaveFile($f, $fmt); $s.Speak($texts[$i]); $s.SetOutputToNull()
    "wrote $f" | Out-File $log -Append -Encoding utf8
  }
  "DONE SAPI $($zh.Name)" | Out-File $log -Append -Encoding utf8
  Write-Host "完成：已用 $($zh.Name) 生成 7 段语音" -ForegroundColor Green
  exit 0
}
# 没有 SAPI 中文语音时，改用 Windows 10/11 的 OneCore 语音
[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
[void][Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
$all = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
"WinRT voices:" | Out-File $log -Append -Encoding utf8
$all | ForEach-Object { "  $($_.DisplayName) | $($_.Language)" } | Out-File $log -Append -Encoding utf8
$v = $all | Where-Object { $_.Language -like 'zh-*' } | Select-Object -First 1
if (-not $v) { "NO_CHINESE_VOICE" | Out-File $log -Append -Encoding utf8; Write-Host "没有找到中文语音，请在 设置-时间和语言-语音 中添加中文语音包" -ForegroundColor Yellow; exit 2 }
$w = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$w.Voice = $v
try { $w.Options.SpeakingRate = 1.1 } catch {}
for ($i = 0; $i -lt $texts.Count; $i++) {
  $stream = Await ($w.SynthesizeTextToStreamAsync($texts[$i])) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $f = Join-Path $dir ("s{0}.wav" -f ($i + 1))
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $n = [uint32]$stream.Size
  Await ($reader.LoadAsync($n)) ([uint32]) | Out-Null
  $bytes = New-Object byte[] $n; $reader.ReadBytes($bytes)
  [IO.File]::WriteAllBytes($f, $bytes)
  "wrote $f" | Out-File $log -Append -Encoding utf8
}
"DONE WINRT $($v.DisplayName)" | Out-File $log -Append -Encoding utf8
Write-Host "完成：已用 $($v.DisplayName) 生成 7 段语音" -ForegroundColor Green
