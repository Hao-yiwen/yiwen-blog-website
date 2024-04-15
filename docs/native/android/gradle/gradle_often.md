# 常用gradle依赖

```kt
implementation(platform("androidx.compose:compose-bom:2023.06.01"))
implementation("androidx.activity:activity-compose:1.7.2")
implementation("androidx.compose.material3:material3")
implementation("androidx.compose.ui:ui")
implementation("androidx.compose.ui:ui-tooling")
implementation("androidx.compose.ui:ui-tooling-preview")
implementation("androidx.core:core-ktx:1.10.1")
implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.1")
// navigation
implementation("androidx.navigation:navigation-compose:2.6.0")
// room
implementation("androidx.room:room-runtime:2.6.1")
ksp("androidx.room:room-compiler:2.6.1")
implementation("androidx.room:room-ktx:2.6.1")
// dataStore
implementation("androidx.datastore:datastore-preferences:1.0.0")
// viewModel
implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.1")
// Retrofit
implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:1.0.0")
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.okhttp3:okhttp:4.11.0")
implementation("io.coil-kt:coil-compose:2.4.0")
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
// Coil 图片懒加载容器
implementation("io.coil-kt:coil-compose:2.4.0")
implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
```