return {
    LrSdkVersion = 11.0,
    LrSdkMinimumVersion = 8.0,

    LrToolkitIdentifier = 'com.superpicky.birdid.lightroom',
    LrPluginName = LOC "$$$/SuperBirdID/Info/PluginName=SuperPicky BirdID Plugin",

    LrInitPlugin = 'PluginInit.lua',

    -- 导出服务（通过导出菜单调用）
    LrExportServiceProvider = {
        {
            title = LOC "$$$/SuperBirdID/Info/ExportTitle=SuperPicky - Bird ID Export",
            file = 'SuperBirdIDExportServiceProvider.lua',
        },
    },

    -- 图库菜单项（通过 图库 → 增效工具 调用）
    LrLibraryMenuItems = {
        {
            title = LOC "$$$/SuperBirdID/Info/MenuTitle=SuperPicky - Identify Current Photo",
            file = 'LibraryMenuItem.lua',
        },
    },

    VERSION = { major=4, minor=0, revision=4, build=1, },
}
