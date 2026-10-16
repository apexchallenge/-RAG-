# -*- coding: utf-8 -*-
"""
用微软 Edge 在线神经语音为视频生成 9 段自然中文旁白（片头 + 七幕 + 片尾）。
只生成音频文件，保存到本脚本所在目录的 voice_natural 文件夹，不修改其他内容。
"""
import asyncio
import os
import sys

try:
    import edge_tts
except ImportError:
    sys.exit("缺少 edge-tts，请先运行：python -m pip install edge-tts")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "voice_natural")

VOICES = {"xiaoxiao": "zh-CN-XiaoxiaoNeural", "yunxi": "zh-CN-YunxiNeural"}

# (文件名, 可用秒数, 旁白)
SEGMENTS = [
    ("s0", 10.45, "这个视频用大约两分钟，讲清楚怎样在不改动数据中台的前提下，接入知识图谱，补上关联查询能力。我们分七步来看。"),
    ("s1", 14.45, "数据中台里，公司、矿山和指标都已经能查。现在业务还想看清它们之间的关联，比如哪家公司运营哪些矿山。前提是，原有数据和业务流程都不改。"),
    ("s2", 19.45, "做法是在中台之外加一套知识图谱，把对象和关系连成一张网。外部同步程序从中台只读获取，只取不改；在外部完成转换和ID对应，再写入Neo4j，一种专门存关系的图数据库，供查询服务和页面使用。"),
    ("s3", 19.45, "同一座矿山，图谱里是M I N 零零零一，中台里是M 零零一。两个ID不必相同，用对应表连起来。程序先按可靠编号和已确认的对应自动匹配；拿不准的标为待核，人工确认后保存，以后直接复用。"),
    ("s4", 19.45, "关系怎么来？中台已有的关系直接保留；字段含义明确的，比如“运营单位”，就按规则生成“运营”关系，并记下来源和审核状态。但要注意：名称相似，不能证明是同一对象；“关联”也不能直接当成“拥有”。"),
    ("s5", 19.45, "查示例甲公司关联矿山二零二四年的锑产量：图谱找到矿山，给出中台对象ID和指标ID；再调用中台原指标接口，也就是原有取数通道，按原统计规则和权限返回数值。缺数显示“暂无数据”，无权限显示“无权限”。"),
    ("s6", 14.45, "新增数据分两种。已有矿山新增年度数值，沿用原对象ID；中台出现新矿山，确认后由外部程序生成新图谱ID并保存对应，有明确依据再建立关系。"),
    ("s7", 9.45, "先用四份CSV表格文件搭练习图谱，验收后换成真实数据，再接入只读接口和联合查询服务。"),
    ("s8", 12.45, "回顾一下：中台只读提供数据，关系保存在外部图谱，数值仍由中台原接口返回。以上为接入方案示意，实际接口需按中台现有能力适配。"),
]


async def synth(text, voice, rate, path):
    """合成一段语音，返回估算时长（秒，来自断句时间戳；拿不到时返回 None）。"""
    comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%")
    end = 0
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                end = max(end, chunk["offset"] + chunk["duration"])
    return end / 1e7 if end else None


async def main():
    log = []
    for key, voice in VOICES.items():
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        print(f"\n== {voice} ==")
        for name, budget, text in SEGMENTS:
            path = os.path.join(d, name + ".mp3")
            rate, dur = 0, None
            for _ in range(4):
                for attempt in range(3):
                    try:
                        dur = await synth(text, voice, rate, path)
                        break
                    except Exception as e:  # 网络偶发错误时重试
                        if attempt == 2:
                            raise
                        print(f"  {name} 重试：{e}")
                        await asyncio.sleep(2)
                if dur is None or dur <= budget * 0.97 or rate >= 30:
                    break
                rate = min(30, rate + int((dur / (budget * 0.97) - 1) * 100) + 2)
            msg = f"{key}/{name}.mp3  语速{rate:+d}%  时长约{dur:.1f}s / 可用{budget}s" if dur else f"{key}/{name}.mp3  语速{rate:+d}%"
            print("  " + msg)
            log.append(msg)
    with open(os.path.join(OUT, "tts_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\nDONE\n")
    print("\n完成！共生成", len(log), "段语音，保存在 voice_natural 文件夹。")


if __name__ == "__main__":
    asyncio.run(main())
