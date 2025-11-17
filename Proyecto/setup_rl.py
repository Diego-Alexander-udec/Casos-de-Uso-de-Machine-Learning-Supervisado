"""
Script para verificar la instalación y entrenar el agente RL
"""

import sys
import subprocess


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("Verificando dependencias...")
    
    required_packages = {
        'gymnasium': 'gymnasium',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'flask': 'Flask'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} instalado correctamente")
        except ImportError:
            print(f"✗ {package_name} NO está instalado")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n⚠️  Paquetes faltantes detectados!")
        print(f"Instalar con: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ Todas las dependencias están instaladas correctamente\n")
    return True


def install_dependencies():
    """Instala las dependencias necesarias."""
    print("\nInstalando dependencias...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "gymnasium", "numpy", "matplotlib", "Flask"
        ])
        print("\n✓ Dependencias instaladas correctamente\n")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Error al instalar dependencias")
        return False


def main():
    print("=" * 70)
    print("CONFIGURACIÓN DEL ENTORNO DE APRENDIZAJE POR REFUERZO")
    print("=" * 70)
    print()
    
    # Verificar dependencias
    if not check_dependencies():
        respuesta = input("\n¿Deseas instalar las dependencias faltantes? (s/n): ")
        if respuesta.lower() == 's':
            if not install_dependencies():
                print("\nNo se pudieron instalar las dependencias.")
                print("Por favor, instálalas manualmente con:")
                print("pip install gymnasium numpy matplotlib Flask")
                sys.exit(1)
        else:
            print("\nNo se pueden ejecutar los scripts sin las dependencias.")
            sys.exit(1)
    
    print("=" * 70)
    print("PASOS SIGUIENTES:")
    print("=" * 70)
    print()
    print("1. Para entrenar el agente RL, ejecuta:")
    print("   python Proyecto/rl_agent_cartpole.py")
    print()
    print("2. Para ver los resultados en la aplicación web:")
    print("   python app.py")
    print("   Luego navega a: http://localhost:5000/caso_practico_refuerzo")
    print()
    print("3. El entrenamiento tomará aproximadamente 2-5 minutos")
    print("   y generará los siguientes archivos:")
    print("   - modelo_rl_cartpole.pkl (modelo entrenado)")
    print("   - static/rl_training_rewards.png (gráficas de recompensas)")
    print("   - static/rl_training_distributions.png (distribuciones)")
    print()
    print("=" * 70)
    print()
    
    # Preguntar si desea entrenar ahora
    respuesta = input("¿Deseas entrenar el agente ahora? (s/n): ")
    if respuesta.lower() == 's':
        print("\nIniciando entrenamiento del agente...")
        print("Por favor espera, esto puede tomar varios minutos...\n")
        
        try:
            import rl_agent_cartpole
            rl_agent_cartpole.main()
            
            print("\n" + "=" * 70)
            print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print("=" * 70)
            print("\nAhora puedes ejecutar la aplicación Flask con:")
            print("python app.py")
            print()
            
        except Exception as e:
            print(f"\n✗ Error durante el entrenamiento: {e}")
            print("\nIntenta ejecutar manualmente:")
            print("python Proyecto/rl_agent_cartpole.py")
    else:
        print("\nPuedes entrenar el agente más tarde con:")
        print("python Proyecto/rl_agent_cartpole.py")


if __name__ == '__main__':
    main()
