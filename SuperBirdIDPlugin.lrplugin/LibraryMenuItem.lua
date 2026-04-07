--[[
    慧眼选鸟 - Library Menu Item
    通过 图库 → 增效工具 菜单快速识别当前选中的照片
]]

local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'
local LrHttp = import 'LrHttp'
local LrView = import 'LrView'
local LrBinding = import 'LrBinding'
local LrFunctionContext = import 'LrFunctionContext'
local LrColor = import 'LrColor'
local LrFileUtils = import 'LrFileUtils'
local LrProgressScope = import 'LrProgressScope'

local LrL10n
local LOC

-- 安全导入 LrL10n（某些 Lightroom 环境可能不支持）
local status, result = pcall(import, 'LrL10n')
if status then
    LrL10n = result
    LOC = LrL10n.loc
else
    -- 降级：从 key 中提取默认文本
    LOC = function(key, ...)
        if type(key) == "string" then
            local val = key:match("=(.*)")
            if val then return val end
        end
        return key
    end
end

-- 插件名称
local PLUGIN_NAME = LOC "$$$/SuperBirdID/PluginName=SuperPicky BirdID"

-- 默认设置
local DEFAULT_API_URL = "http://127.0.0.1:5156"
local DEFAULT_TOP_K = 3

-- Unicode解码辅助函数
local function decodeUnicodeEscape(str)
    if not str then return str end

    local function unicodeToUtf8(code)
        code = tonumber(code, 16)
        if code < 0x80 then
            return string.char(code)
        elseif code < 0x800 then
            return string.char(
                0xC0 + math.floor(code / 0x40),
                0x80 + (code % 0x40)
            )
        elseif code < 0x10000 then
            return string.char(
                0xE0 + math.floor(code / 0x1000),
                0x80 + (math.floor(code / 0x40) % 0x40),
                0x80 + (code % 0x40)
            )
        end
        return "?"
    end

    return str:gsub("\\u(%x%x%x%x)", unicodeToUtf8)
end

-- 简单的JSON解析函数
local function parseJSON(jsonString)
    local result = {}

    local success = string.match(jsonString, '"success"%s*:%s*([^,}]+)')
    if success then
        result.success = (success == "true")
    end

    local resultsBlock = string.match(jsonString, '"results"%s*:%s*%[(.-)%]')
    if resultsBlock then
        result.results = {}

        for itemBlock in string.gmatch(resultsBlock, '{([^{}]*)}') do
            local item = {}

            local cn_name_raw = string.match(itemBlock, '"cn_name"%s*:%s*"([^"]*)"')
            local en_name_raw = string.match(itemBlock, '"en_name"%s*:%s*"([^"]*)"')
            local display_name_raw = string.match(itemBlock, '"display_name"%s*:%s*"([^"]*)"')
            local sci_name_raw = string.match(itemBlock, '"scientific_name"%s*:%s*"([^"]*)"')
            local desc_raw = string.match(itemBlock, '"description"%s*:%s*"([^"]*)"')

            item.cn_name = decodeUnicodeEscape(cn_name_raw)
            item.en_name = decodeUnicodeEscape(en_name_raw)
            item.display_name = decodeUnicodeEscape(display_name_raw) or item.cn_name  -- 兼容旧版本 API
            item.scientific_name = decodeUnicodeEscape(sci_name_raw)
            item.description = decodeUnicodeEscape(desc_raw)

            local confStr = string.match(itemBlock, '"confidence"%s*:%s*([%d%.]+)')
            item.confidence = confStr and tonumber(confStr) or 0

            if item.display_name or item.cn_name then
                table.insert(result.results, item)
            end
        end
    end

    local error_raw = string.match(jsonString, '"error"%s*:%s*"([^"]*)"')
    result.error = decodeUnicodeEscape(error_raw)

    local warning_raw = string.match(jsonString, '"warning"%s*:%s*"([^"]*)"')
    result.warning = decodeUnicodeEscape(warning_raw)

    return result
end

-- 简单的JSON编码函数
local function encodeJSON(tbl)
    local parts = {}
    for k, v in pairs(tbl) do
        local key = '"' .. tostring(k) .. '"'
        local value
        if type(v) == "string" then
            value = '"' .. v:gsub('"', '\\"'):gsub('\\', '\\\\') .. '"'
        elseif type(v) == "boolean" then
            value = tostring(v)
        elseif type(v) == "number" then
            value = tostring(v)
        else
            value = '"' .. tostring(v) .. '"'
        end
        table.insert(parts, key .. ":" .. value)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

-- 识别照片
local function recognizePhoto(photo, apiUrl)
    local photoPath = photo:getRawMetadata("path")
    local photoName = photo:getFormattedMetadata("fileName") or "Unknown"

    if not LrFileUtils.exists(photoPath) then
        return {
            success = false,
            error = LOC("$$$/SuperBirdID/Error/FileNotFound=File not found: ^1", photoName),
            photoName = photoName
        }
    end

    local requestBody = encodeJSON({
        image_path = photoPath,
        use_yolo = true,
        use_gps = true,
        top_k = DEFAULT_TOP_K
    })

    local response, headers = LrHttp.post(
        apiUrl .. "/recognize",
        requestBody,
        {
            { field = "Content-Type", value = "application/json" }
        }
    )

    if not response then
        return {
            success = false,
            error = LOC("$$$/SuperBirdID/Error/ApiFailed=API Call Failed"),
            photoName = photoName
        }
    end

    local result = parseJSON(response)
    result.photoName = photoName
    result.photo = photo

    return result
end

-- 保存识别结果到照片元数据
-- 写入 Title (鸟名) 和 Caption (描述)
-- V4.0.5: 使用服务端返回的 display_name，不再自行判断语言
--   服务端已根据 SuperPicky 主程序语言设置选择中文/英文名称
local function saveRecognitionResult(photo, displayName, description)
    local catalog = LrApplication.activeCatalog()

    catalog:withWriteAccessDo(LOC "$$$/SuperBirdID/Undo/SaveResult=Save Bird Recognition Result", function()
        -- Title: 使用服务端已选好的 display_name
        photo:setRawMetadata("title", displayName)
        -- Caption: 写入描述（如有）
        if description and description ~= "" then
            photo:setRawMetadata("caption", description)
        end
    end)
end

-- 显示结果选择对话框
local function showResultSelectionDialog(results, photoName)
    local selectedIndex = nil

    LrFunctionContext.callWithContext("resultSelectionDialog", function(context)
        local f = LrView.osFactory()
        local props = LrBinding.makePropertyTable(context)

        props.selectedBird = 1

        local candidateViews = {}

        for i, bird in ipairs(results) do
            local confidence = bird.confidence or 0
            local displayName = bird.display_name or bird.cn_name or LOC "$$$/SuperBirdID/Dialog/Unknown=Unknown"
            local enName = bird.en_name or ""
            
            local confColor
            if confidence >= 50 then
                confColor = LrColor(0.2, 0.7, 0.3)
            elseif confidence >= 20 then
                confColor = LrColor(0.8, 0.6, 0.1)
            else
                confColor = LrColor(0.6, 0.6, 0.6)
            end

            candidateViews[#candidateViews + 1] = f:row {
                spacing = f:control_spacing(),
                
                f:radio_button {
                    title = "",
                    value = LrView.bind { key = 'selectedBird', object = props },
                    checked_value = i,
                    width = 20,
                },
                
                f:column {
                    spacing = 2,
                    
                    f:row {
                        f:static_text {
                            title = string.format("%d.", i),
                            font = "<system/bold>",
                            width = 20,
                        },
                        f:static_text {
                            title = displayName,
                            font = "<system/bold>",
                        },
                        f:static_text {
                            title = string.format("  %.1f%%", confidence),
                            text_color = confColor,
                            font = "<system/bold>",
                        },
                    },
                    
                    f:static_text {
                        title = "    " .. enName,
                        text_color = LrColor(0.5, 0.5, 0.5),
                        font = "<system/small>",
                    },
                },
            }
            
            if i < #results then
                candidateViews[#candidateViews + 1] = f:spacer { height = 6 }
                candidateViews[#candidateViews + 1] = f:separator { fill_horizontal = 1 }
                candidateViews[#candidateViews + 1] = f:spacer { height = 6 }
            end
        end

        candidateViews[#candidateViews + 1] = f:spacer { height = 12 }
        candidateViews[#candidateViews + 1] = f:separator { fill_horizontal = 1 }
        candidateViews[#candidateViews + 1] = f:spacer { height = 8 }
        
        candidateViews[#candidateViews + 1] = f:row {
            f:radio_button {
                title = "",
                value = LrView.bind { key = 'selectedBird', object = props },
                checked_value = 0,
                width = 20,
            },
            f:static_text {
                title = LOC "$$$/SuperBirdID/Dialog/Skip=Skip photo, do not write",
                text_color = LrColor(0.5, 0.5, 0.5),
            },
        }

        local candidatesGroup = f:column(candidateViews)

        local dialogContent = f:column {
            spacing = f:control_spacing(),
            fill_horizontal = 1,

            f:spacer { width = 350 },

            f:row {
                f:static_text {
                    title = LOC("$$$/SuperBirdID/Dialog/ResultTitle=Result for ^1", photoName),
                    font = "<system/bold>",
                },
            },
            
            f:spacer { height = 8 },
            f:separator { fill_horizontal = 1 },
            f:spacer { height = 12 },

            candidatesGroup,
            
            f:spacer { height = 8 },
        }

        local dialogResult = LrDialogs.presentModalDialog({
            title = PLUGIN_NAME,
            contents = dialogContent,
            actionVerb = LOC "$$$/SuperBirdID/Dialog/Action=Write EXIF",
            cancelVerb = LOC "$$$/SuperBirdID/Dialog/Cancel=Cancel",
            resizable = true,
        })

        if dialogResult == "ok" then
            selectedIndex = props.selectedBird
        else
            selectedIndex = nil
        end
    end)

    return selectedIndex
end

-- 主函数
LrTasks.startAsyncTask(function()
    local catalog = LrApplication.activeCatalog()
    local targetPhoto = catalog:getTargetPhoto()

    -- 检查是否选中了照片
    if not targetPhoto then
        LrDialogs.message(PLUGIN_NAME,
            LOC "$$$/SuperBirdID/Message/SelectOne=Please select one photo first",
            "warning")
        return
    end

    -- 检查API服务是否可用
    local healthCheck = LrHttp.get(DEFAULT_API_URL .. "/health")

    if not healthCheck or string.find(healthCheck, '"status"%s*:%s*"ok"') == nil then
        LrDialogs.message(PLUGIN_NAME,
            LOC "$$$/SuperBirdID/Message/ApiError=Cannot connect to SuperPicky API Service\n\nPlease ensure:\n1. SuperPicky App is running\n2. BirdID API Service is started",
            "error")
        return
    end

    -- 检查是否选中了多张照片
    local selectedPhotos = catalog:getTargetPhotos()
    if #selectedPhotos > 1 then
        -- 批量处理模式
        local progressScope = LrProgressScope({
            title = LOC "$$$/SuperBirdID/Progress/BatchTitle=Identifying Birds (Batch Mode)"
        })
        progressScope:setCancelable(true)
        
        local successCount = 0
        local failCount = 0
        local total = #selectedPhotos
        
        for i, photo in ipairs(selectedPhotos) do
            if progressScope:isCanceled() then break end
            
            progressScope:setPortionComplete(i - 1, total)
            local fileName = photo:getFormattedMetadata("fileName")
            progressScope:setCaption(string.format("%s (%d/%d)", fileName, i, total))
            
            local result = recognizePhoto(photo, DEFAULT_API_URL)
            
            if result.success and result.results and #result.results > 0 then
                -- 自动选择第一个结果 (置信度最高)
                local best = result.results[1]
                local displayName = best.display_name or best.cn_name or best.en_name or "Unknown"
                saveRecognitionResult(photo, displayName, best.description)
                successCount = successCount + 1
            else
                failCount = failCount + 1
            end
        end
        
        progressScope:done()
        
        local msg = string.format("Batch processing completed.\nTotal: %d\nSuccess: %d\nFailed: %d", total, successCount, failCount)
        LrDialogs.message(PLUGIN_NAME, msg, "info")
        return
    end

    -- 单张处理模式
    local result = recognizePhoto(targetPhoto, DEFAULT_API_URL)

    if result.success and result.results and #result.results > 0 then
        -- GPS 回退时提示用户
        if result.warning then
            LrDialogs.message(
                PLUGIN_NAME,
                result.warning,
                "info"
            )
        end
        -- 显示结果选择对话框
        local selectedIndex = showResultSelectionDialog(result.results, result.photoName)

        if selectedIndex and selectedIndex > 0 then
            local selectedBird = result.results[selectedIndex]
            local displayName = selectedBird.display_name or selectedBird.cn_name or selectedBird.en_name or "Unknown"

            saveRecognitionResult(targetPhoto, displayName, selectedBird.description)
        end
    else
        local errorMsg
        if result.error then
            errorMsg = result.error
        elseif result.success then
            -- success=true 但 results=[] 表示 YOLO 未检测到鸟
            errorMsg = LOC "$$$/SuperBirdID/Dialog/NoBirdDetected=No bird detected in this photo"
        else
            errorMsg = LOC "$$$/SuperBirdID/Dialog/Unknown=Unknown error"
        end

        local failMsg = LOC("$$$/SuperBirdID/Message/IdentifyFail=Cannot identify bird in this photo\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nError:\n^1\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nPossible reasons:\n• No bird or bird unclear\n• File corrupted or unsupported\n• Model not loaded", errorMsg)

        LrDialogs.message(PLUGIN_NAME .. " - " .. LOC("$$$/SuperBirdID/Dialog/IdentifyFailTitle=Identify Failed"), failMsg, "error")
    end
end)
