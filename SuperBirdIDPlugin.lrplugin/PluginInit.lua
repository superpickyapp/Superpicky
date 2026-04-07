local LrLogger = import 'LrLogger'

local myLogger = LrLogger( 'SuperBirdIDPlugin' )
myLogger:enable( "logfile" )

myLogger:info( "慧眼选鸟 Lightroom 插件初始化完成 - v4.0.0" )