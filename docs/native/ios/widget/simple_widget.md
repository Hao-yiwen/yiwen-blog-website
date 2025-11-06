---
title: 小组件构成部分
sidebar_label: 小组件构成部分
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 小组件构成部分

一个 iOS 小组件通常由以下三部分组成：

1.	Widget: 这是小组件的主要入口，定义了小组件的标识符和配置。
2.	WidgetConfigurationIntent: 这是可选的配置意图，允许用户自定义小组件的行为或外观。
3.	TimelineProvider: 这是时间线提供者，用于生成并提供小组件的数据和时间线条目。

## 代码示例

```swift
import WidgetKit
import SwiftUI
import Intents
import AppIntents

struct ConfigurationAppIntent: WidgetConfigurationIntent {
    static var title: LocalizedStringResource = "Configuration"
    static var description = IntentDescription("This is an example widget.")

    @Parameter(title: "Favorite Emoji", default: "😃")
    var favoriteEmoji: String
}

struct Provider: TimelineProvider {
    func placeholder(in context: Context) -> SimpleEntry {
        SimpleEntry(date: Date(), configuration: ConfigurationAppIntent())
    }

    func getSnapshot(in context: Context, completion: @escaping (SimpleEntry) -> ()) {
        let entry = SimpleEntry(date: Date(), configuration: ConfigurationAppIntent())
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SimpleEntry>) -> ()) {
        var entries: [SimpleEntry] = []

        let currentDate = Date()
        for hourOffset in 0 ..< 5 {
            let entryDate = Calendar.current.date(byAdding: .hour, value: hourOffset, to: currentDate)!
            let entry = SimpleEntry(date: entryDate, configuration: ConfigurationAppIntent())
            entries.append(entry)
        }

        let timeline = Timeline(entries: entries, policy: .atEnd)
        completion(timeline)
    }
}

struct SimpleEntry: TimelineEntry {
    let date: Date
    let configuration: ConfigurationAppIntent
}

struct WidgetEntryView: View {
    var entry: Provider.Entry

    var body: some View {
        Text(entry.date, style: .time)
            .containerBackground(Color.blue, for: .widget)
    }
}

@main
struct MyWidget: Widget {
    let kind: String = "MyWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: Provider()) { entry in
            WidgetEntryView(entry: entry)
        }
        .configurationDisplayName("My Widget")
        .description("This is an example widget.")
    }
}

struct MyWidget_Previews: PreviewProvider {
    static var previews: some View {
        WidgetEntryView(entry: SimpleEntry(date: Date(), configuration: ConfigurationAppIntent()))
            .previewContext(WidgetPreviewContext(family: .systemSmall))
    }
}

```