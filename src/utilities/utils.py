import subprocess


def download_landcover_files(config):
    """
    Download landcover files from s3 given the config file
    """
    DataPath = "/home/ubuntu/ExtData"

    # conditionally set variables to create s3 and landcover file paths
    gridDir = (
        f"{config['Res']}_{config['RegionID']}" if len(config["RegionID"]) == 2 else ""
    )

    if config["Met"] == "geosfp":
        metUC = "GEOSFP"
        metDir = "GEOS_FP"
        constYr = "2011"
        LandCoverFileExtension = "nc"
    elif config["Met"] == "merra2":
        metUC = "MERRA2"
        metDir = metUC
        constYr = "2015"
        LandCoverFileExtension = "nc4"

    if config["Res"] == "4x5":
        gridRes = "${Res}"
    elif config["Res"] == "2x2.5":
        gridRes = "2x25"
    elif config["Res"] == "0.5x0.625":
        gridRes = "05x0625"
    elif config["Res"] == "0.25x0.3125":
        gridRes = "025x03125"

    if len(config["RegionID"]) == 2:
        LandCoverFile = f"{DataPath}/GEOS_{gridDir}/{metDir}/{constYr}/01/{metUC}.{constYr}0101.CN.{gridRes}.{config['RegionID']}.{LandCoverFileExtension}"
        s3_lc_path = f"s3://gcgrid/GEOS_{gridDir}/{metDir}/{constYr}/01/{metUC}.{constYr}0101.CN.{gridRes}.{config['RegionID']}.{LandCoverFileExtension}"
    else:
        LandCoverFile = f"{DataPath}/GEOS_{gridDir}/{metDir}/{constYr}/01/{metUC}.{constYr}0101.CN.{gridRes}.{LandCoverFileExtension}"
        s3_lc_path = f"s3://gcgrid/GEOS_{gridDir}/{metDir}/{constYr}/01/{metUC}.{constYr}0101.CN.{gridRes}.{LandCoverFileExtension}"

    # run the aws command to download the files
    command = f"aws s3 cp --request-payer=requester {s3_lc_path} {LandCoverFile}"
    results = subprocess.run(command.split(), capture_output=True, text=True)

    output = (
        "Successfully downloaded landcover files"
        if results.returncode == 0
        else results
    )
    print(output)
