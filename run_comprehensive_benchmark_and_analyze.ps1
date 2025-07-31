# PowerShell script for comprehensive LLM-ATC-HAL benchmark and analysis
# run_comprehensive_benchmark_and_analyze.ps1

param(
    [int]$NumHorizontal = 500,
    [int]$NumVertical = 500,
    [int]$NumSector = 500,
    [string]$Complexities = "simple,moderate,complex",
    [string]$ShiftLevels = "baseline,mild,extreme",
    [int]$Horizon = 10,
    [int]$MaxInterventions = 3,
    [double]$StepSize = 10.0,
    [string]$OutputDir = "",
    [string]$AnalysisFormat = "comprehensive",
    [switch]$SkipBenchmark,
    [switch]$SkipAnalysis,
    [switch]$UseNewCLI
)

Write-Host "🚀 Starting Comprehensive LLM-ATC-HAL Benchmark and Analysis" -ForegroundColor Green

# Generate output directory if not provided
if ([string]::IsNullOrEmpty($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "experiments/comprehensive_long_run_$timestamp"
}

Write-Host "📁 Output Directory: $OutputDir" -ForegroundColor Cyan

# Calculate total scenarios
$complexityArray = $Complexities -split ','
$shiftArray = $ShiftLevels -split ','
$totalScenarios = ($NumHorizontal + $NumVertical + $NumSector) * $complexityArray.Count * $shiftArray.Count

Write-Host "📊 Total Scenarios: $totalScenarios" -ForegroundColor Yellow
Write-Host "   - Horizontal: $NumHorizontal × $($complexityArray.Count) × $($shiftArray.Count) = $($NumHorizontal * $complexityArray.Count * $shiftArray.Count)" -ForegroundColor White
Write-Host "   - Vertical: $NumVertical × $($complexityArray.Count) × $($shiftArray.Count) = $($NumVertical * $complexityArray.Count * $shiftArray.Count)" -ForegroundColor White
Write-Host "   - Sector: $NumSector × $($complexityArray.Count) × $($shiftArray.Count) = $($NumSector * $complexityArray.Count * $shiftArray.Count)" -ForegroundColor White

$success = $true

# Method 1: Use the new CLI with auto-analyze feature
if ($UseNewCLI -and -not $SkipBenchmark) {
    Write-Host "🔄 Running benchmark with integrated analysis..." -ForegroundColor Blue
    
    $cmd = "python cli.py monte-carlo-benchmark " +
           "--num-horizontal $NumHorizontal " +
           "--num-vertical $NumVertical " +
           "--num-sector $NumSector " +
           "--complexities `"$Complexities`" " +
           "--shift-levels `"$ShiftLevels`" " +
           "--horizon $Horizon " +
           "--max-interventions $MaxInterventions " +
           "--step-size $StepSize " +
           "--output-dir `"$OutputDir`" " +
           "--auto-analyze " +
           "--analysis-format `"$AnalysisFormat`""
    
    Write-Host "Command: $cmd" -ForegroundColor Gray
    Invoke-Expression $cmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Integrated benchmark and analysis failed" -ForegroundColor Red
        $success = $false
    } else {
        Write-Host "✅ Integrated benchmark and analysis completed successfully!" -ForegroundColor Green
    }
} else {
    # Method 2: Run benchmark and analysis separately
    
    # Step 1: Run Monte Carlo Benchmark
    if (-not $SkipBenchmark) {
        Write-Host "🔄 Running Monte Carlo Benchmark..." -ForegroundColor Blue
        
        $benchmarkCmd = "python cli.py monte-carlo-benchmark " +
                       "--num-horizontal $NumHorizontal " +
                       "--num-vertical $NumVertical " +
                       "--num-sector $NumSector " +
                       "--complexities `"$Complexities`" " +
                       "--shift-levels `"$ShiftLevels`" " +
                       "--horizon $Horizon " +
                       "--max-interventions $MaxInterventions " +
                       "--step-size $StepSize " +
                       "--output-dir `"$OutputDir`""
        
        Write-Host "Command: $benchmarkCmd" -ForegroundColor Gray
        Invoke-Expression $benchmarkCmd
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Benchmark failed - aborting" -ForegroundColor Red
            $success = $false
            exit 1
        }
    } else {
        Write-Host "⏭️  Skipping benchmark" -ForegroundColor Yellow
    }

    # Step 2: Run Analysis
    if (-not $SkipAnalysis) {
        Write-Host "🔍 Running Results Analysis..." -ForegroundColor Blue
        
        # Check if results directory exists
        if (-not (Test-Path $OutputDir)) {
            Write-Host "❌ Results directory not found: $OutputDir" -ForegroundColor Red
            exit 1
        }
        
        $analysisCmd = "python cli.py analyze " +
                      "--results-dir `"$OutputDir`" " +
                      "--output-format `"$AnalysisFormat`""
        
        Write-Host "Command: $analysisCmd" -ForegroundColor Gray
        Invoke-Expression $analysisCmd
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Analysis failed" -ForegroundColor Red
            $success = $false
        }
    } else {
        Write-Host "⏭️  Skipping analysis" -ForegroundColor Yellow
    }
}

# Summary
if ($success) {
    Write-Host "🎉 Comprehensive benchmark and analysis completed successfully!" -ForegroundColor Green
    Write-Host "📁 Results available in: $OutputDir" -ForegroundColor Cyan
    
    # List generated files
    if (Test-Path $OutputDir) {
        Write-Host "📄 Generated files:" -ForegroundColor Yellow
        Get-ChildItem -Path $OutputDir -Recurse -File | ForEach-Object {
            $relativePath = $_.FullName.Replace((Resolve-Path $OutputDir).Path, "").TrimStart('\')
            Write-Host "   - $relativePath" -ForegroundColor White
        }
    }
} else {
    Write-Host "❌ Some operations failed - check output above" -ForegroundColor Red
    exit 1
}

Write-Host "✅ All operations completed!" -ForegroundColor Green
