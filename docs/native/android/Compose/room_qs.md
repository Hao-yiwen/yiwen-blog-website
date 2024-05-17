# fallbackToDestructiveMigration问题

## 场景

```kt
@Database(
    entities = [Airport::class, Favorite::class, FavoriteContainer::class],
    version = 4,
    exportSchema = false
)
abstract class FlightDataBase : RoomDatabase() {
    abstract fun airportDao(): AirportDao;

    abstract fun favoriteContainerDao(): FavoriteContainerDao;

    companion object {
        @Volatile
        private var Instance: FlightDataBase? = null

        fun getDataBase(context: Context): FlightDataBase {
            return Instance ?: synchronized(this) {
                Room.databaseBuilder(context, FlightDataBase::class.java, "app_database")
                    .createFromAsset("flight_search.db")
                    /**
                     * @exception 我的场景是根据flight_search.db创建数据库，然后再根据dao创建一个表，但是使用fallbackToDestructiveMigration后每次
                     * 打开app都会导致应用数据库重构，所以在这里关闭数据库，如果在开发阶段需要重构数据库，可以打开这个注释，
                     * 目前问题并没有得到定位，但是fallbackToDestructiveMigration方法确实要少用。。。。
                     */
//                    .fallbackToDestructiveMigration()
                    .build().also { Instance = it }
            }
        }
    }
}
```

我现在有个场景，在app初始化的时候从flight_search.db创建Airport和Favorite两个表，然后FavoriteContainer表是根据entity自己创建的，并不需要初始化。

## 问题

在这种场景下，每次杀死app重启app，在不改变数据库的版本的情况下也会导致数据库重建，原有所有数据会消失。我在想这是不是和FavoriteContainer表是根据entity自己创建的有关，目前先记录这个问题，后面再排查，后续使用fallbackToDestructiveMigration方法需要多注意。