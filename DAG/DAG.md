```mermaid
graph TD
    %% ==========================================
    %% LAYER 0: CONFOUNDERS
    %% ==========================================
    subgraph Confounders
        Theta_i["Đặc tính cá nhân θ_i<br/>(heterogeneity)"] 
        ToD["Thời gian trong ngày<br/>TimeOfDay"]
        Leverage["Mức đòn bẩy<br/>Leverage"]
        PortfolioLimit["Giới hạn vị thế<br/>PortfolioLimit_MM"]
    end

    Theta_i --> Sigma_t
    Theta_i --> GammaMM_t
    Theta_i --> GammaMom_t
    Theta_i --> Tau_t
    Theta_i --> Ithr_t

    ToD --> Sigma_t
    ToD --> Vol_t

    Leverage --> GammaMom_t
    Leverage --> OrderSize_t

    PortfolioLimit --> GammaMM_t
    PortfolioLimit --> QuoteDepth_t

    %% ==========================================
    %% LAYER 1: EXOGENOUS & PAST
    %% ==========================================
    News_t["News_t"] --> Uncert_t["Uncertainty_t"]
    Vol_tminus1["Volatility_{t-1}"] --> Uncert_t
    BaseStress["BaselineStress_i"] --> Sigma_t["LatentStress_t"]

    BaseGammaMM["BaselineRiskAversion_MM"] --> GammaMM_t
    BaseGammaMom["BaselineRiskAversion_Mom"] --> GammaMom_t

    %% ==========================================
    %% LAYER 2: STRESS → BEHAVIOR
    %% ==========================================
    Sigma_t --> GammaMM_t
    Sigma_t --> GammaMom_t
    Sigma_t --> Tau_t["ReactionTime_t"]
    Sigma_t --> Ithr_t["InattentionThreshold_t"]
    Sigma_t --> Part_t["LiquidityParticipation_t"]

    %% ==========================================
    %% LAYER 2b: INATTENTION LOGIC
    %% ==========================================
    subgraph InattentionLogic
        SignalInput["Tín hiệu thị trường"] --> Check{"Xác suất chú ý<br/>f(σ, Ithr_t)"}
        Check -->|Bỏ qua| Drop["Drop signal → No trade"]
        Check -->|Chú ý| SigProc_t2["SignalProcessing_t2 → Trade"]
    end

    %% ==========================================
    %% LAYER 3: INVENTORY & MM DYNAMICS
    %% ==========================================
    OF_tminus1["OrderFlow_{t-1}"] --> Inv_t["Inventory_MM_t"]
    Inv_t --> GammaMM_t
    Inv_t --> Inv_tplus["Inventory_MM_{t+1}"]

    GammaMM_t --> QuoteSpread_t["QuoteSpread_t"]
    GammaMM_t --> QuoteDepth_t["QuoteDepth_t"]

    GammaMom_t --> OrderSize_t["OrderSize_t"]
    Tau_t --> OrderTiming_t["OrderTiming_t"]
    
    SigProc_t2 --> OF_t["OrderFlow_t"]   
    Part_t --> OrderArrival_t["OrderArrival_t"]

    %% ==========================================
    %% LAYER 4: MARKET & DEPTH 
    %% ==========================================
    %% QuoteDepth tác động Depth và Spread
    QuoteDepth_t --> Spread_t["BidAskSpread_t"]
    QuoteDepth_t -->|adds| Depth_tplus["Depth_{t+1}"]
    
    QuoteSpread_t --> Spread_t
    Depth_t["Depth_t"] --> Spread_t
    Depth_t --> Depth_tplus

    %% THÊM MỚI (ĐIỂM SỐ 4): Spread rộng làm giảm lượng lệnh mới nhảy vào thị trường
    Spread_t --> OrderArrival_t

    OrderArrival_t --> OF_t
    OrderSize_t --> OF_t
    OrderTiming_t --> OF_t
    
    OrderSize_t -->|consumes| Depth_tplus
    
    OF_t --> Midprice_t["Midprice_t"]

    %% ==========================================
    %% LAYER 5: RETURNS & VOLATILITY
    %% ==========================================
    Midprice_t --> Returns_t["Returns_t"]
    Returns_t --> Vol_t["Volatility_t"]

    %% ==========================================
    %% LAYER 6: FEEDBACK LOOP
    %% ==========================================
    Vol_t --> Uncert_tplus["Uncertainty_{t+1}"]
    Uncert_tplus --> Sigma_tplus["LatentStress_{t+1}"]
    
    Sigma_tplus --> GammaMM_tplus["GammaMM_{t+1}"]
    Sigma_tplus --> GammaMom_tplus["GammaMom_{t+1}"]

    %% ==========================================
    %% STYLE DEFINITIONS & COLOR CODING
    %% ==========================================
    classDef conf fill:#d4f8d4,stroke:#2e7d32,stroke-width:2px;
    classDef stress fill:#ffd6d6,stroke:#c62828,stroke-width:2px;
    classDef action fill:#d6e4ff,stroke:#1565c0,stroke-width:2px;
    classDef market fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef output fill:#fff3cd,stroke:#f9a825,stroke-width:2px;
    classDef attention fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px;

    class Theta_i,ToD,Leverage,PortfolioLimit conf
    class Sigma_t,Sigma_tplus,Uncert_t,Uncert_tplus,BaseStress,News_t stress
    class GammaMM_t,GammaMom_t,GammaMM_tplus,GammaMom_tplus,Tau_t,Ithr_t,Part_t,OrderSize_t,OrderTiming_t,OrderArrival_t,BaseGammaMM,BaseGammaMom action
    class SignalInput,Check,Drop,SigProc_t2 attention
    class Depth_t,Depth_tplus,QuoteSpread_t,QuoteDepth_t,Spread_t,Inv_t,Inv_tplus,OF_t,OF_tminus1,Midprice_t market
    class Returns_t,Vol_t,Vol_tminus1 output
    ```