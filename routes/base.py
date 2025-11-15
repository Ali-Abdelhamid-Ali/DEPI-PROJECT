from fastapi import APIRouter , Depends
from helpers.configs import Settings , get_settings


base_router = APIRouter(
    prefix=f"/{get_settings().APP_NAME}",
    tags=["DevTune"]) 

@base_router.get("/") 
async def welcome(app_settings: Settings = Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    
    return {"app_name": app_name, "app_version": app_version}

@base_router.get("/health") 
async def health(app_settings: Settings = Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    return {"app_name": app_name, "app_version": app_version , "status": "healthy"}