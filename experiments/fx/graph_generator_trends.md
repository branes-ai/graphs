#    TorchScript vs FX Comparison:
    
    TorchScript:
    + Mature and stable
    + Good for production deployment
    + Can serialize entire models
    + Works with complex control flow (via scripting)
    - Less flexible for analysis and transformation
    - Harder to debug
    - Being phased out for new features
    
    FX (torch.fx):
    + Modern, actively developed
    + Excellent for graph analysis and transformation
    + Better debugging and introspection
    + More Pythonic and easier to work with
    + Better integration with newer PyTorch features
    - May struggle with very complex Python code
    - Newer, so less production-tested
    
    RECOMMENDATION: Use FX for new development, especially for:
    - Model analysis and optimization
    - Custom transformations
    - Research and experimentation
    
    Use TorchScript only when you need:
    - Production deployment to C++ environments
    - Legacy system compatibility

    
  # Step 2
    
    
    Classical MPC Results:
    ✓ Successfully drives plant to setpoint (MAE: {mae:.2f}°C)
    ✓ Handles setpoint changes and disturbances
    ✓ Respects physical constraints
    
    Graph Analysis Results:
    ✓ TorchScript: Captures complete computational flow
    ✓ FX: Provides detailed node-by-node analysis
    ✓ Both methods reveal optimization opportunities
    
    Key Insights:
    • MPC involves complex optimization loops (not captured in graphs)
    • Neural enhancement adds differentiable components
    • FX provides better introspection for development
    • TorchScript better for production deployment
    
    Recommendation for C++ Development:
    • Use FX for model analysis and optimization
    • Export to TorchScript for C++ integration
    • Consider libtorch for full MPC implementation
    • Implement optimization loop in C++ for best performance


    
    