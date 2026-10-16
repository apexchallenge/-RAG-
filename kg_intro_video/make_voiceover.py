#!/usr/bin/env python3
"""
给《数据中台只读接入知识图谱》无声版视频合成中文配音。

两种用法：
  1) 自动配音（需要能访问网络）：
       pip install edge-tts
       python make_voiceover.py
     edge-tts 调用的是微软 Edge 浏览器的在线朗读服务；商用场景建议改用正式的云语音服务。

  2) 使用自己的录音：
       把七段录音命名为 s1.wav ... s7.wav（mp3/m4a 也可以）放进一个文件夹，然后
       python make_voiceover.py --from-dir 录音文件夹

依赖：ffmpeg / ffprobe（需在 PATH 中）。
输出：kg_intro_voiced.mp4（画面不重新编码，只替换音轨）。
"""
import argparse
import asyncio
import math
import os
import subprocess
import sys
import tempfile

# 各幕起止时间（秒）与旁白，和视频时间轴一一对应
SCENES = [
    (0, 15, "数据中台里，公司、矿山和指标都已经能查。现在业务还想看清它们之间的关联，比如哪家公司运营哪些矿山。前提是，原有数据和业务流程都不改。"),
    (15, 35, "做法是在中台之外加一套知识图谱，把对象和关系连成一张网。外部同步程序从中台只读获取，只取不改；在外部完成转换和ID对应，再写入Neo4j，一种专门存关系的图数据库，供查询服务和页面使用。"),
    (35, 55, "同一座矿山，图谱里是M-I-N零零零一，中台里是M零零一。两个ID不必相同，用对应表连起来。程序先按可靠编号和已确认的对应自动匹配；拿不准的标为待核，人工确认后保存，以后直接复用。"),
    (55, 75, "关系怎么来？中台已有的关系直接保留；字段含义明确的，比如“运营单位”，就按规则生成“运营”关系，并记下来源和审核状态。但要注意：名称相似，不能证明是同一对象；“关联”也不能直接当成“拥有”。"),
    (75, 95, "查示例甲公司关联矿山二零二四年的锑产量：图谱找到矿山，给出中台对象ID和指标ID；再调用中台原指标接口，也就是原有取数通道，按原统计规则和权限返回数值。缺数显示“暂无数据”，无权限显示“无权限”。"),
    (95, 110, "新增数据分两种。已有矿山新增年度数值，沿用原对象ID；中台出现新矿山，确认后由外部程序生成新图谱ID并保存对应，有明确依据再建立关系。"),
    (110, 120, "先用四份CSV表格文件搭练习图谱，验收后换成真实数据，再接入只读接口和联合查询服务。"),
]
LEAD = 0.3   # 每幕开始后多久开口
TAIL = 0.25  # 每幕结束前留白


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"命令失败：{' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).strip())


async def synth(text, voice, rate, out):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(out)


def tts_scene(i, text, budget, voice, tmp):
    out = os.path.join(tmp, f"s{i}.mp3")
    rate = 0
    for _ in range(4):
        asyncio.run(synth(text, voice, f"{rate:+d}%", out))
        d = duration(out)
        if d <= budget or rate >= 40:
            break
        rate = min(40, rate + math.ceil((d / budget - 1) * 100) + 2)
    print(f"  第{i}幕：{d:.1f}s / 可用 {budget:.1f}s（语速 {rate:+d}%）")
    return out


def fit(path, budget, tmp, i):
    """录音超出该幕时长时，轻微加速塞进时间窗（最多 1.35 倍）。"""
    trimmed = os.path.join(tmp, f"trim{i}.wav")
    sr = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-af", f"{sr},areverse,{sr},areverse", "-ar", "48000", "-ac", "2", trimmed])
    path = trimmed
    d = duration(path)
    out = os.path.join(tmp, f"fit{i}.wav")
    speed = max(1.0, min(1.35, d / budget))
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-af", f"atempo={speed:.4f}", "-ar", "48000", "-ac", "2", out])
    print(f"  第{i}幕：录音 {d:.1f}s，可用 {budget:.1f}s，{'加速 %.2f 倍' % speed if speed > 1 else '无需加速'}")
    if d / budget > 1.35:
        print(f"  ⚠ 第{i}幕录音 {d:.1f}s 明显超出 {budget:.1f}s，结尾可能与下一幕重叠，建议重录得更紧凑些")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="kg_intro_silent.mp4")
    ap.add_argument("--out", default="kg_intro_voiced.mp4")
    ap.add_argument("--voice", default="zh-CN-YunxiNeural", help="如 zh-CN-YunxiNeural（男声）/ zh-CN-XiaoxiaoNeural（女声）")
    ap.add_argument("--from-dir", help="使用自己的录音：文件夹内放 s1~s7 的 wav/mp3/m4a")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (t0, t1, text) in enumerate(SCENES, 1):
            budget = (t1 - t0) - LEAD - TAIL
            if a.from_dir:
                src = next((os.path.join(a.from_dir, f"s{i}{ext}") for ext in (".wav", ".mp3", ".m4a") if os.path.exists(os.path.join(a.from_dir, f"s{i}{ext}"))), None)
                if not src:
                    sys.exit(f"找不到第{i}幕录音：{a.from_dir}/s{i}.wav")
            else:
                src = tts_scene(i, text, budget, a.voice, tmp)
            parts.append((t0 + LEAD, fit(src, budget, tmp, i)))

        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", a.video]
        for _, p in parts:
            cmd += ["-i", p]
        delays = "".join(f"[{k + 1}:a]adelay={int(st * 1000)}|{int(st * 1000)}[a{k}];" for k, (st, _) in enumerate(parts))
        mix = "".join(f"[a{k}]" for k in range(len(parts))) + f"amix=inputs={len(parts)}:normalize=0,apad,loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
        cmd += ["-filter_complex", delays + mix, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart", a.out]
        run(cmd)
    print(f"完成：{a.out}")


if __name__ == "__main__":
    main()
